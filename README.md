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

## Installation

The project requires the following dependencies:
  - Python 3.10
  - PyTorch 2.0.1+
  - CUDA 18
  - NumPy 1.23.5
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
pip install numpy==1.23.5
pip install language_tool_python PyMultiDictionary tqdm wandb gdown tiktoken dataclasses datasets 
pip install torch==2.0.1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```
## Datasets

The multilingual Backpack LM is trained on the following datasets:
  - [Europarl](https://www.statmt.org/europarl/): Parallel French-English corpus from the European Parliament proceedings.
  - [MultiUN](https://opus.nlpl.eu/MultiUN/en&fr/v1/MultiUN): Parallel corpus extracted from United Nations documents.
To download these datasets, run:

```bash:
cd data
bash get_bash.sh
```

To tokenize these datasets and preprocess for training, run:

```bash:
cd data
bash europarl/prepare.py
bash multiun/prepare.py
```

## Training

We save the checkpoint of different models trained on Europarl and MultiUN on Google Drive:



To train a Backpack LM model, follow these steps:
1. Configure the training setup:
  - Modify the configuration file in config/ to set up the training parameters (e.g., `model_name`, `wandb_log`, `learning_rate`, `device`).
2. Train the model:
  - Start training with the following command:
```bash
python3.10 train.py config/train_small_16.py --out_dir=out-bkp-small-16 --model_name=backpack-lm
```
  - Resume a training with following command:
```bash
python3.10 train.py config/train_small_16.py --out_dir=out-bkp-small-16 --model_name=backpack-lm --init_from=resume
```

## Evaluation

The evaluation includes both intrinsic and extrinsic metrics:
  - Perplexity: Assesses the model’s ability to predict held-out text.
```bash
python3.10 perplexity_per_lang.py config/train_mini_16.py --model_name=backpack-lm --out_dir=out-bkp-mini-16 --device=cuda
```
  - Cloze task: Measures the model’s accuracy in filling in missing words.
```bash
python3.10 sense_visualisation.py --model_name=backpack-lm --out_dir=out-bkp-small-16 --device=cuda
```
  - Sense visualization and sense distribution: Analyzes the learned sense vectors for word representation.
```bash
python3.10 cloze_test.py --model_name=backpack-lm --out_dir=out-bkp-small-16 --device=cuda
```
