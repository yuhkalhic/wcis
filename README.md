# WCIS Project

## Introduction
**What Can I Say? Enhancing Transformer Architectures with an Improved LoRA Method Using Continuous State-Space Equations for Efficient Convergence.**
Welcome to the WCIS project repository. This project focuses on enhancing transformer architectures by implementing an improved LoRA (Low-Rank Adaptation) method. Our approach utilizes continuous state-space equations to ensure efficient convergence, significantly enhancing the model's performance without a considerable increase in computational costs.

## Features
- Advanced implementation of LoRA layers in a transformer model for sequence classification.
- Utilization of PyTorch, Transformers, and PyTorch Lightning frameworks for efficient model training and evaluation.
- Highly configurable model parameters through command-line arguments, allowing fine-tuning and experimentation.

## Installation

### Prerequisites
- Anaconda or Miniconda is recommended for environment management.
- Git for cloning the repository.

### Clone the Repository
To start using the WCIS project, clone the repository to your local machine:
```bash
git clone https://github.com/yuhkalhic/wcis.git
cd wcis
conda create --name wcis python=3.8
conda activate wcis
pip install -r requirements.txt
python src/train_model.py --lora_alpha 32 --lora_r 16

