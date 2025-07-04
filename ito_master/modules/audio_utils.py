# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import typing as tp

import numpy as np
import julius
import torch
import torchaudio


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.
    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels.
    """
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 12,
                       energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.
    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        output (torch.Tensor): Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    # transform = torchaudio.transforms.Loudness(sample_rate)
    # print(f"wav shape: {wav.shape}")
    # input_loudness_db = transform(wav).item()
    input_loudness_db = torchaudio.functional.loudness(wav, sample_rate)
    if torch.isnan(input_loudness_db).any():
        return None
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    if len(wav.shape)==3:
        gain = gain.reshape(gain.shape[0], 1, 1)
    output = gain * wav
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    log_clipping: bool = False, sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).
    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (Optional[str]): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    else:
        assert wav.dtype == torch.int16
        return wav.float() / 2**15


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.
    ..Warning:: There exist many formula for doing this convertion. None are perfect
    due to the asymetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistancies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav


def generate_pink_noise(duration, sample_rate=44100, scale=1.0):
    """
    Generate pink noise.
    
    Args:
    duration (float): Duration of the noise in seconds.
    sample_rate (int): Sample rate in Hz. Default is 44100 Hz.

    Returns:
    numpy.ndarray: Pink noise signal.
    """
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white_noise = np.random.normal(0, 1, num_samples)
    
    # Generate pink noise in frequency domain
    freqs = np.fft.rfftfreq(num_samples)
    
    # Create pink noise spectrum (avoiding division by zero)
    pink_spectrum = np.where(freqs == 0, 1, 1 / np.sqrt(freqs))
    
    # Apply pink spectrum to white noise
    white_noise_fft = np.fft.rfft(white_noise)
    pink_noise_fft = white_noise_fft * pink_spectrum
    
    # Convert back to time domain
    pink_noise = np.fft.irfft(pink_noise_fft)
    
    # Normalize
    pink_noise = pink_noise / np.max(np.abs(pink_noise))
    pink_noise *= scale
    
    return pink_noise






if __name__ == "__main__":
    import os
    import soundfile as sf

    duration = 5  # 5 seconds
    sample_rate = 44100  # Standard CD-quality sample rate

    # Create output directory if it doesn't exist
    output_dir = "pink_noise_samples"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(3):
        # Generate pink noise
        pink_noise = generate_pink_noise(duration, sample_rate)

        # Convert to float32 for soundfile
        pink_noise = pink_noise.astype(np.float32)

        # Save the audio file
        filename = os.path.join(output_dir, f"pink_noise_sample_{i+1}.wav")
        sf.write(filename, pink_noise, sample_rate)
        print(f"Generated and saved: {filename}")