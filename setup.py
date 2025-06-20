# setup.py
from setuptools import setup, find_packages

setup(
    name="ito_master",
    version="0.1.1",
    packages=find_packages(),
    install_requires = [
        "torch>=2.0",
        "torchaudio==2.1.2",
        "soundfile",
        "librosa",
        "scipy",
        "numba",
        "auraloss",
        "dasp-pytorch",
        "torchlpc==0.5",
        "torchcomp",
        "pytorch-lightning",
        "julius",
        "pyloudnorm",
        "matplotlib",
        "pymixconsole",
        "soxbindings",
        "aubio",
        "laion-clap",
        "torchvision",
        "huggingface_hub",
    ],
    author="junghyun-tony-koo-sony",
    description="ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors",
    long_description_content_type="text/markdown",
    url="https://github.com/SonyResearch/ITO-Master",
    author_email='junghyun.koo@sony.com', 
    license='CC BY-NC 4.0'
)
