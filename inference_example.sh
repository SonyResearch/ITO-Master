#!/bin/bash
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU
gpu_stat="--inference_device cuda"
# gpu_stat='--inference_device cpu'

paths="--input_path examples/input_1.flac --reference_path examples/reference_1.flac"

white_box="--model_type white_box"
black_box="--model_type black_box"

### Style Transfer
# white-box
python inference.py $paths $white_box $gpu_stat --output_dir_path outputs/white_box_st/
# black-box
python inference.py $paths $black_box $gpu_stat --output_dir_path outputs/black_box_st/

### Style Transfer + ITO
ito_config="--ito_reference_path examples/reference_1.flac --perform_ito --num_steps 100 --ito_save_freq 10 --learning_rate 0.01"
af_config="--ito_objective AudioFeatureLoss"
clap_aud_config="--ito_objective CLAPFeatureLoss --clap_target_type Audio"
text_prompt="heavy metal"
clap_txt_config=(--ito_objective CLAPFeatureLoss --clap_target_type Text --clap_text_prompt "${text_prompt}")

# white-box, AF Loss
python inference.py $ito_config $af_config $paths $white_box $gpu_stat --output_dir_path outputs/white_box_ito_af/
# white-box, CLAP Audio Loss
python inference.py $ito_config $clap_aud_config $paths $white_box $gpu_stat --output_dir_path outputs/white_box_ito_clapaud/
# white-box, CLAP Text Loss
python inference.py $ito_config "${clap_txt_config[@]}" $paths $white_box $gpu_stat --output_dir_path outputs/white_box_ito_claptxt/

# black-box, AF Loss
python inference.py $ito_config $af_config $paths $black_box $gpu_stat --output_dir_path outputs/black_box_ito_af/
# black-box, CLAP Audio Loss
python inference.py $ito_config $clap_aud_config $paths $black_box $gpu_stat --output_dir_path outputs/black_box_ito_clapaud/
# black-box, CLAP Text Loss
python inference.py $ito_config "${clap_txt_config[@]}" $paths $black_box $gpu_stat --output_dir_path outputs/black_box_ito_claptxt/
