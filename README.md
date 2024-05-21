# WCIS Project

## Introduction
**What Can I Say? Enhancing Transformer Architectures with an Improved LoRA Method Using Continuous State-Space Equations for Efficient Convergence.**


### Quick start
To start using the WCIS project, clone the repository to your local machine:
```bash
git clone https://github.com/yuhkalhic/wcis.git
cd wcis
conda create --name wcis python=3.8
conda activate wcis
pip install -r requirements.txt
python src/train_model.py --lora_alpha 32 --lora_r 16

