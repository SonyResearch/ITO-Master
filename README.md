# ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/spaces/jhtonyKoo/ITO-Master)
[![Audio Samples](https://img.shields.io/badge/🎧-Audio_Samples-blue.svg)](https://tinyurl.com/ITO-Master)

This repository contains the official implementation of the paper:

> **"ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors"**  
> Junghyun Koo, Marco A. Martínez-Ramírez, Wei-Hsiang Liao, Giorgio Fabbro, Michele Mancusi, Yuki Mitsufuji  
> Presented at **ISMIR 2025**

---

## 🔧 Installation

```bash
sudo apt-get update && sudo apt-get install -y \
  libsox-fmt-all \
  libsox-dev \
  sox \
  libsndfile1
pip install ito_master
```

---

## 🚀 Inference Examples
Basic usage examples (full examples available in [`inference_example.sh`](./inference_example.sh)):

### 🎛️ Style Transfer Only

```bash
python inference.py \
  --input_path examples/input_1.flac \
  --reference_path examples/reference_1.flac \
  --model_type white_box \
  --inference_device cuda \
  --output_dir_path outputs/white_box_st/
```

### 🎛️ Style Transfer + ITO (AudioFeatureLoss)

```bash
python inference.py \
  --input_path examples/input_1.flac \
  --reference_path examples/reference_1.flac \
  --model_type white_box \
  --inference_device cuda \
  --perform_ito \
  --ito_reference_path examples/reference_1.flac \
  --ito_objective AudioFeatureLoss \
  --num_steps 100 \
  --ito_save_freq 10 \
  --learning_rate 0.01 \
  --output_dir_path outputs/white_box_ito_af/
```

### 🎛️ Style Transfer + ITO (CLAPFeatureLoss with text prompt)

```bash
python inference.py \
  --input_path examples/input_1.flac \
  --reference_path examples/reference_1.flac \
  --model_type white_box \
  --inference_device cuda \
  --perform_ito \
  --ito_reference_path examples/reference_1.flac \
  --ito_objective CLAPFeatureLoss \
  --clap_target_type Text \
  --clap_text_prompt "heavy metal" \
  --num_steps 100 \
  --ito_save_freq 10 \
  --learning_rate 0.01 \
  --output_dir_path outputs/white_box_ito_claptxt/
```

---

## 📜 Citation

Please cite our work if you find it useful:

```bibtex
@INPROCEEDINGS{koo2025, 
  author={Koo, Junghyun and Martínez-Ramírez, Marco A. and Liao, Wei-Hsiang and Fabbro, Giorgio and Mancusi, Michele and Mitsufuji, Yuki}, 
  booktitle={The 26th International Society for Music Information Retrieval Conference (ISMIR)},  
  title={ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors}, 
  year={2025},
}
```
