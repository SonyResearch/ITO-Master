"""
    Inference script of the paper "ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors".
     - published at International Society for Music Information Retrieval (ISMIR) 2025
    
    This implementation belongs to Sony Research
        Repo link: https://github.com/SonyResearch/ITO-Master
"""

import torch
import soundfile as sf
import numpy as np
import argparse
import os
import yaml
import julius
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from ito_master import Dasp_Mastering_Style_Transfer, Effects_Encoder, TCNModel, AudioFeatureLoss, CLAPFeatureLoss, Audio_Effects_Normalizer, lufs_normalize


class MasteringStyleTransfer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu" if args.inference_device=="cpu" else f"cuda")
        # Load ckpt files from HuggingFace Hub if not provided
        self.args.encoder_path = hf_hub_download(
            repo_id="jhtonyKoo/ITO-Master",
            repo_type="space",
            filename="models/mastering_effects_encoder.pt" if self.args.model_type == "white_box" else "models/mastering_effects_encoder_blackbox.pt",
        )
        self.args.model_path = hf_hub_download(
            repo_id="jhtonyKoo/ITO-Master",
            repo_type="space",
            filename="models/white_box_converter.pt" if self.args.model_type == "white_box" else "models/black_box_converter.pt",
        )
        # Load models
        self.effects_encoder = self.load_effects_encoder()
        print("\t-> Effects Encoder loaded successfully")
        self.mastering_converter = self.load_mastering_converter()
        print("\t-> Mastering Converter loaded successfully")
        self.fx_normalizer = Audio_Effects_Normalizer(precomputed_feature_path=args.fx_norm_feature_path, \
                                                        STEMS=['mixture'], \
                                                        EFFECTS=['eq', 'imager', 'loudness'])
        # Loss functions
        if args.ito_objective=='AudioFeatureLoss':
            self.af_loss = AudioFeatureLoss(
                weights=ito_config['af_weights'],
                sample_rate=ito_config['sample_rate'],
                stem_separation=False,
                use_clap=False
            )
        elif args.ito_objective == 'CLAPFeatureLoss':
            self.clap_loss = CLAPFeatureLoss()

    def load_effects_encoder(self):
        effects_encoder = Effects_Encoder(self.args.cfg_enc)
        reload_weights(effects_encoder, self.args.encoder_path, self.device)
        effects_encoder.to(self.device)
        effects_encoder.eval()
        return effects_encoder

    def load_mastering_converter(self):
        if self.args.model_type == "white_box":
            mastering_converter = Dasp_Mastering_Style_Transfer(num_features=2048,
                                                                sample_rate=self.args.sample_rate,
                                                                tgt_fx_names=['eq', 'distortion', 'multiband_comp', 'gain', 'imager', 'limiter'],
                                                                model_type='tcn',
                                                                config=self.args.cfg_converter,
                                                                batch_size=1)
        elif self.args.model_type == "black_box":
            mastering_converter = TCNModel(nparams=args.cfg_converter["condition_dimension"], \
                                        ninputs=2, \
                                        noutputs=2, \
                                        nblocks=args.cfg_converter["nblocks"], \
                                        dilation_growth=args.cfg_converter["dilation_growth"], \
                                        kernel_size=args.cfg_converter["kernel_size"], \
                                        channel_width=args.cfg_converter["channel_width"], \
                                        stack_size=args.cfg_converter["stack_size"], \
                                        cond_dim=args.cfg_converter["condition_dimension"], \
                                        causal=args.cfg_converter["causal"])
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
        reload_weights(mastering_converter, self.args.model_path, self.device)
        mastering_converter.to(self.device)
        mastering_converter.eval()
        return mastering_converter

    def get_reference_embedding(self, reference_tensor):
        with torch.no_grad():
            reference_feature = self.effects_encoder(reference_tensor)
        return reference_feature

    def mastering_style_transfer(self, input_tensor, reference_feature):
        # pad at the front and back for black-box model
        if self.args.model_type == "black_box":
            input_tensor = torch.nn.functional.pad(input_tensor, (1024, 1024), mode='reflect')
        # perform style transfer - output shape: (num_channels, num_samples)
        output_audio = self.mastering_converter(input_tensor, reference_feature)
        if self.args.model_type == "white_box":
            predicted_params = self.mastering_converter.get_last_predicted_params()
        else:
            predicted_params = None
            output_audio = output_audio[..., 1024:-1024]  # remove padding for black-box model
        return output_audio, predicted_params

    def inference_time_optimization(self, input_tensor, reference_path, ito_config, initial_reference_feature):
        # load ITO reference audio
        if not(ito_config['ito_objective'] == 'CLAPFeatureLoss' and ito_config['clap_target_type'] == 'Text'):
            reference_audio = sf.read(reference_path)
            reference_tensor = self.preprocess_audio(reference_audio, self.args.sample_rate)

        fit_embedding = torch.nn.Parameter(initial_reference_feature, requires_grad=True)
        optimizer = getattr(torch.optim, ito_config['optimizer'])([fit_embedding], lr=ito_config['learning_rate'])

        min_loss = float('inf')
        min_loss_step = 0
        all_results = []

        for step in tqdm(range(ito_config['num_steps'])):
            optimizer.zero_grad()
            
            # Style transfer forward
            output_audio, current_params = self.mastering_style_transfer(input_tensor, fit_embedding)

            # Compute loss
            if ito_config['ito_objective'] == 'AudioFeatureLoss':
                losses = self.af_loss(output_audio, reference_tensor)
                total_loss = sum(losses.values())
            elif ito_config['ito_objective'] == 'CLAPFeatureLoss':
                if ito_config['clap_target_type'] == 'Audio':
                    target = reference_tensor
                else:
                    target = ito_config['clap_text_prompt']
                total_loss = self.clap_loss(output_audio, target, self.args.sample_rate, distance_fn=ito_config['clap_distance_fn'])

            total_loss.backward()
            optimizer.step()

            if total_loss < min_loss:
                min_loss = total_loss.item()
                min_loss_step = step

            # Log top 5 parameter differences
            if step == 0:
                initial_params = current_params
            top_5_diff = self.get_top_n_diff_string(initial_params, current_params, top_n=5)
            log_entry = f"Step {step + 1}\n   Loss: {total_loss.item():.4f}\n{top_5_diff}\n"

            output_audio = output_audio[0].T.detach().cpu().numpy()
            if self.args.loudness_norm_output:
                output_audio = lufs_normalize(output_audio, self.args.sample_rate, lufs=-14.0)
            all_results.append({
                'step': step + 1,
                'loss': total_loss.item(),
                'audio': output_audio,
                'params': current_params,
                'log': log_entry
            })

        return all_results, min_loss_step

    def preprocess_audio(self, audio, target_sample_rate=44100, normalize=False):
        data, sample_rate = audio

        # Ensure stereo channels
        if data.ndim == 1:
            data = np.stack([data, data])
        elif data.ndim == 2:
            if data.shape[0] == 2:
                pass  # Already in correct shape
            elif data.shape[1] == 2:
                data = data.T
            else:
                data = np.stack([data[:, 0], data[:, 0]])  # Duplicate mono channel
        else:
            raise ValueError(f"Unsupported audio shape: {data.shape}")

        # Resample if necessary
        if sample_rate != target_sample_rate:
            data = julius.resample_frac(torch.from_numpy(data), sample_rate, target_sample_rate).numpy()

        # Apply fx normalization for input audio during mastering style transfer
        if normalize:
            data = self.fx_normalizer.normalize_audio(data.T, 'mixture').T

        # Convert to torch tensor
        data_tensor = torch.FloatTensor(data).unsqueeze(0)

        return data_tensor.to(self.device)

    def process_audio(self, input_audio, reference_audio):
        # load audio and preprocess
        input_audio, reference_audio = sf.read(input_audio), sf.read(reference_audio)
        input_tensor = self.preprocess_audio(input_audio, self.args.sample_rate, normalize=True)
        reference_tensor = self.preprocess_audio(reference_audio, self.args.sample_rate)
        # Extract reference feature
        reference_feature = self.get_reference_embedding(reference_tensor)
        # Perform style transfer
        with torch.no_grad():
            output_audio, predicted_params = self.mastering_style_transfer(input_tensor, reference_feature)
        output_audio = output_audio[0].T.detach().cpu().numpy()
        if self.args.loudness_norm_output:
            output_audio = lufs_normalize(output_audio, self.args.sample_rate, lufs=-14.0)

        return output_audio, predicted_params, self.args.sample_rate, input_tensor, reference_feature

    def get_param_output_string(self, params):
        if params is None:
            return "No parameters available"
        
        param_mapper = {
            'eq': {
                'low_shelf_gain_db': ('Low Shelf Gain', 'dB', -20, 20),
                'low_shelf_cutoff_freq': ('Low Shelf Cutoff', 'Hz', 20, 2000),
                'low_shelf_q_factor': ('Low Shelf Q', '', 0.1, 5.0),
                'band0_gain_db': ('Low-Mid Band Gain', 'dB', -20, 20),
                'band0_cutoff_freq': ('Low-Mid Band Frequency', 'Hz', 80, 2000),
                'band0_q_factor': ('Low-Mid Band Q', '', 0.1, 5.0),
                'band1_gain_db': ('Mid Band Gain', 'dB', -20, 20),
                'band1_cutoff_freq': ('Mid Band Frequency', 'Hz', 2000, 8000),
                'band1_q_factor': ('Mid Band Q', '', 0.1, 5.0),
                'band2_gain_db': ('High-Mid Band Gain', 'dB', -20, 20),
                'band2_cutoff_freq': ('High-Mid Band Frequency', 'Hz', 8000, 12000),
                'band2_q_factor': ('High-Mid Band Q', '', 0.1, 5.0),
                'band3_gain_db': ('High Band Gain', 'dB', -20, 20),
                'band3_cutoff_freq': ('High Band Frequency', 'Hz', 12000, 20000),
                'band3_q_factor': ('High Band Q', '', 0.1, 5.0),
                'high_shelf_gain_db': ('High Shelf Gain', 'dB', -20, 20),
                'high_shelf_cutoff_freq': ('High Shelf Cutoff', 'Hz', 4000, 20000),
                'high_shelf_q_factor': ('High Shelf Q', '', 0.1, 5.0),
            },
            'distortion': {
                'drive_db': ('Drive', 'dB', 0, 8),
                'parallel_weight_factor': ('Dry/Wet Mix', '%', 0, 100),
            },
            'multiband_comp': {
                'low_cutoff': ('Low/Mid Crossover', 'Hz', 20, 1000),
                'high_cutoff': ('Mid/High Crossover', 'Hz', 1000, 20000),
                'parallel_weight_factor': ('Dry/Wet Mix', '%', 0, 100),
                'low_shelf_comp_thresh': ('Low Band Comp Threshold', 'dB', -60, 0),
                'low_shelf_comp_ratio': ('Low Band Comp Ratio', ': 1', 1, 20),
                'low_shelf_exp_thresh': ('Low Band Exp Threshold', 'dB', -60, 0),
                'low_shelf_exp_ratio': ('Low Band Exp Ratio', ': 1', 1, 20),
                'low_shelf_at': ('Low Band Attack Time', 'ms', 5, 100),
                'low_shelf_rt': ('Low Band Release Time', 'ms', 5, 100),
                'mid_band_comp_thresh': ('Mid Band Comp Threshold', 'dB', -60, 0),
                'mid_band_comp_ratio': ('Mid Band Comp Ratio', ': 1', 1, 20),
                'mid_band_exp_thresh': ('Mid Band Exp Threshold', 'dB', -60, 0),
                'mid_band_exp_ratio': ('Mid Band Exp Ratio', ': 1', 0, 1),
                'mid_band_at': ('Mid Band Attack Time', 'ms', 5, 100),
                'mid_band_rt': ('Mid Band Release Time', 'ms', 5, 100),
                'high_shelf_comp_thresh': ('High Band Comp Threshold', 'dB', -60, 0),
                'high_shelf_comp_ratio': ('High Band Comp Ratio', ': 1', 1, 20),
                'high_shelf_exp_thresh': ('High Band Exp Threshold', 'dB', -60, 0),
                'high_shelf_exp_ratio': ('High Band Exp Ratio', ': 1', 1, 20),
                'high_shelf_at': ('High Band Attack Time', 'ms', 5, 100),
                'high_shelf_rt': ('High Band Release Time', 'ms', 5, 100),
            },
            'gain': {
                'gain_db': ('Output Gain', 'dB', -24, 24),
            },
            'imager': {
                'width': ('Stereo Width', '', 0, 1),
            },
            'limiter': {
                'threshold': ('Threshold', 'dB', -60, 0),
                'at': ('Attack Time', 'ms', 5, 100),
                'rt': ('Release Time', 'ms', 5, 100),
            },
        }
        
        output = []
        for fx_name, fx_params in params.items():
            output.append(f"{fx_name.upper()}:")
            if isinstance(fx_params, dict):
                for param_name, param_value in fx_params.items():
                    if isinstance(param_value, torch.Tensor):
                        param_value = param_value.item()
                    
                    if fx_name in param_mapper and param_name in param_mapper[fx_name]:
                        friendly_name, unit, min_val, max_val = param_mapper[fx_name][param_name]
                        if unit=='%':
                            param_value = param_value * 100
                        current_content = f"  {friendly_name}: {param_value:.2f} {unit}"
                        if param_name=='mid_band_exp_ratio':
                            current_content += f" (Range: {min_val}-{max_val})"
                        output.append(current_content)
                    else:
                        output.append(f"  {param_name}: {param_value:.2f}")
            else:
                # stereo imager
                width_percentage = fx_params.item() * 200
                output.append(f"  Stereo Width: {width_percentage:.2f}% (Range: 0-200%)")
    
        return "\n".join(output)

    def get_top_n_diff_string(self, initial_params, ito_params, top_n=5):
        if initial_params is None or ito_params is None:
            return "Cannot compare parameters"
        
        all_diffs = []
        for fx_name in initial_params.keys():
            if isinstance(initial_params[fx_name], dict):
                for param_name in initial_params[fx_name].keys():
                    initial_value = initial_params[fx_name][param_name]
                    ito_value = ito_params[fx_name][param_name]
                    
                    param_range = self.mastering_converter.fx_processors[fx_name].param_ranges[param_name]
                    normalized_diff = abs((ito_value - initial_value) / (param_range[1] - param_range[0]))
                    
                    all_diffs.append((fx_name, param_name, initial_value.item(), ito_value.item(), normalized_diff.item()))
            else:
                initial_value = initial_params[fx_name]
                ito_value = ito_params[fx_name]
                normalized_diff = abs(ito_value - initial_value)
                all_diffs.append((fx_name, 'width', initial_value.item(), ito_value.item(), normalized_diff.item()))

        top_diffs = sorted(all_diffs, key=lambda x: x[4], reverse=True)[:top_n]
        
        output = [f"   Top {top_n} parameter differences (initial / ITO / normalized diff):"]
        for fx_name, param_name, initial_value, ito_value, normalized_diff in top_diffs:
            output.append(f"      {fx_name.upper()} - {param_name}: {initial_value:.2f} / {ito_value:.2f} / {normalized_diff:.2f}")
        
        return "\n".join(output)

def reload_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k.replace('module.', '') if k.startswith('module.') else k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mastering Style Transfer")
    data_args = parser.add_argument_group('Data args')
    data_args.add_argument("--input_path", type=str, required=True, help="Path to input audio file")
    data_args.add_argument("--reference_path", type=str, required=True, help="Path to reference audio file")
    data_args.add_argument("--sample_rate", type=int, default=44100, help="Sampling rate")
    data_args.add_argument("--output_dir_path", type=str, default='outputs/', help="Directory to save output audio files")
    data_args.add_argument("--output_wav_name", type=str, default='output_style_transferred', help="Name of the output audio file")
    model_args = parser.add_argument_group('Model args')
    model_args.add_argument("--model_type", type=str, default="white_box", choices=["white_box", "black_box"], help="Model type for mastering style transfer converter")
    model_args.add_argument("--path_to_config", type=str, default='ito_master/networks/configs.yaml', help="Path to network architecture configuration file")
    model_args.add_argument("--fx_norm_feature_path", type=str, default='fxnorm_feat.npy', help="Path to Fx Normalization feature statistics file")
    ito_args = parser.add_argument_group('ITO args')
    ito_args.add_argument("--perform_ito", action="store_true", help="Whether to perform ITO")
    ito_args.add_argument("--ito_objective", type=str, default="AudioFeatureLoss", choices=['AudioFeatureLoss', 'CLAPFeatureLoss'], help="which objective to use for ITO, AudioFeatureLoss or CLAPFeatureLoss")
    ito_args.add_argument("--ito_reference_path", type=str, required=False, help="Path to ITO reference audio file")
    ito_args.add_argument("--clap_target_type", type=str, choices=['Audio', 'Text'], help="which modality to use for CLAP, Audio or Text")
    ito_args.add_argument("--clap_text_prompt", type=str, required=False, help="Text prompt for ITO on CLAP embeddings")
    hyperparam_args = parser.add_argument_group('Hyperparameters args')
    hyperparam_args.add_argument("--optimizer", type=str, default="RAdam", help="Optimizer for ITO")
    hyperparam_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for ITO")
    hyperparam_args.add_argument("--num_steps", type=int, default=100, help="Number of optimization steps for ITO")
    hyperparam_args.add_argument("--af_weights", nargs='+', type=float, default=[0.1, 0.001, 1.0, 1.0, 0.1], help="Weights for AudioFeatureLoss")
    hyperparam_args.add_argument("--clap_distance_fn", type=str, default='cosine', choices=['cosine', 'l1', 'mse'], help="which distance function to use for CLAP: cosine, l1 or mse")
    hyperparam_args.add_argument("--ito_save_freq", type=int, default=10, help="Frequency of saving ITO results")
    hyperparam_args.add_argument("--loudness_norm_output", action="store_true", default=True, help="Whether to apply loudness normalization to the output audio")
    inference_args = parser.add_argument_group('Inference args')
    inference_args.add_argument('--inference_device', type=str, default='cuda', help="if this option is not set to 'cpu', inference will happen on gpu only if there is a detected one")
    args = parser.parse_args()

    # load network configurations
    with open(args.path_to_config, 'r') as f:
        configs = yaml.full_load(f)
    args.cfg_converter = configs['TCN']['param_mapping']
    args.cfg_enc = configs['Effects_Encoder']['default']

    ito_config = {
        'ito_objective': args.ito_objective,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'num_steps': args.num_steps,
        'af_weights': args.af_weights,
        'sample_rate': args.sample_rate,
        'clap_text_prompt': args.clap_text_prompt,
        'clap_target_type': args.clap_target_type,
        'clap_distance_fn': args.clap_distance_fn
    }

    mastering_style_transfer = MasteringStyleTransfer(args)
    ''' Style Transfer '''
    print("Starting Style Transfer...")
    output_audio, predicted_params, sample_rate, input_tensor, initial_reference_feature = mastering_style_transfer.process_audio(
        args.input_path, args.reference_path
    )
    # Save results
    os.makedirs(args.output_dir_path, exist_ok=True)
    # Save the output audio
    sf.write(os.path.join(args.output_dir_path, f"{args.output_wav_name}_init.wav"), output_audio, sample_rate)
    # Save the predicted parameters
    if args.model_type == "white_box":
        param_output = mastering_style_transfer.get_param_output_string(predicted_params)
        with open(os.path.join(args.output_dir_path, "predicted_params_init.txt"), 'w') as f:
            f.write("Predicted Parameters (Initial Style Transferred Output):\n")
            f.write(param_output)
    print("\t-> Style Transfer completed and results saved.")

    ''' Inference-Time Optimization '''
    if args.perform_ito:
        print("Starting Inference-Time Optimization...")
        all_results, min_loss_step = mastering_style_transfer.inference_time_optimization(
            input_tensor, args.ito_reference_path, ito_config, initial_reference_feature
        )
        # Save results
        cur_output_dir = os.path.join(args.output_dir_path, "ito_results")
        os.makedirs(cur_output_dir, exist_ok=True)
        # save loss plot
        losses = [result['loss'] for result in all_results]
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Loss over ITO Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(os.path.join(cur_output_dir, f"{args.output_wav_name}_loss_plot.png"))
        plt.close()

        # save best output audio
        best_output_audio = all_results[min_loss_step]['audio']
        sf.write(os.path.join(args.output_dir_path, f"{args.output_wav_name}_ito_best.wav"), best_output_audio, args.sample_rate)

        # save all results
        if args.ito_save_freq!=0:
            if args.model_type == "white_box":
                # initialize param text file
                with open(os.path.join(cur_output_dir, "predicted_params.txt"), 'w') as f:
                    f.write("Predicted Parameters at Initial Style Transfer Output:\n")
                    f.write(mastering_style_transfer.get_param_output_string(predicted_params) + "\n\n")
                f.close()
                with open(os.path.join(cur_output_dir, "log.txt"), 'w') as f:
                    f.write("Inference-Time Optimization Log:\n")
                f.close()
            for cur_save_step in range(args.ito_save_freq - 1, len(all_results), args.ito_save_freq):
                cur_result = all_results[cur_save_step]
                sf.write(os.path.join(cur_output_dir, f"output_step#{cur_save_step+1}.wav"), cur_result['audio'], args.sample_rate)
                if args.model_type == "white_box":
                    # save predicted parameters
                    with open(os.path.join(cur_output_dir, f"predicted_params.txt"), 'a') as f:
                        f.write(f"Predicted Parameters at Step {cur_save_step+1}:\n")
                        f.write(mastering_style_transfer.get_param_output_string(cur_result['params']) + "\n\n")
                    f.close()
                    # save log
                    with open(os.path.join(cur_output_dir, "log.txt"), 'a') as f:
                        f.write(cur_result['log'])
                    f.close()

            print(f"\t-> ITO completed with minimum loss at step {min_loss_step + 1}. Results saved to {cur_output_dir}.")
