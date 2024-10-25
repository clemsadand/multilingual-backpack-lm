# Multilingual Backpack Language Models

This repository contains the code, data, and experiments for the Multilingual Backpack Language Model, a project aimed at extending Backpack LMs to multilingual settings. Backpack LMs provide a flexible interface for interpretability and control in language modeling by explicitly encoding multiple senses for words. This work explores training Backpack LMs on parallel French-English corpora to efficiently handle polysemy in multilingual contexts.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Backpack LMs learn multiple sense vectors per word, allowing for explicit modeling of polysemous words. Previously tested in monolingual settings for English and Chinese, this project extends the Backpack architecture to multilingual modeling by training on both English and French using Europarl and MultiUN datasets. The multilingual Backpack LM efficiently encodes word meanings across languages, demonstrating lower perplexity and improved accuracy on cloze tasks compared to baseline GPT-2 models.

## Project Structure

The repository is organized as follows:


## Installation

The project requires the following dependencies:
- Python 3.10
- PyTorch 2.0.1+
- Hugging Face Transformers
- NumPy
- pandas
- matplotlib
- wandb
- tiktoken
- datasets
- dataclasses
- PyMultiDictionary
- language_tool_python
- tqdm
- gdown
- cuda

1. Clone this repository.

```bash
git clone https://github.com/clemsadand/multilingual-backpack-lm.git
cd working_dir
```

2. You need to install NVIDIA-drivers. Run:

```bash
cd bkp_install
bash bkp_nvidia.sh 
```

3. You may need to install anaconda or miniconda. To install miniconda, run:

```bash
cd bkp_install
bash anaconda.sh
```

4. You need to create a virtual environment with Python3.10.

- With conda:
  
```bash
conda create --name bkp python=3.10
conda activate bkp # to activate
```

- Without conda:

```bash
python3.10 -m venv bkp
source bkp/bin/activate #to activate
```

5. To install the required packages, run:

```bash
pip install language_tool_python PyMultiDictionary tqdm wandb gdown tiktoken dataclasses datasets 
pip install torch==2.0.1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```
## Datasets
