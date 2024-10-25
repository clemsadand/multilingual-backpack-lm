# Multilingual Backpack Language Models

This repository contains the code, data, and experiments for the Multilingual Backpack Language Model, a project aimed at extending Backpack LMs to multilingual settings. Backpack LMs provide a flexible interface for interpretability and control in language modeling by explicitly encoding multiple senses for words. This work explores training Backpack LMs on parallel French-English corpora to efficiently handle polysemy in multilingual contexts.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)


## Introduction

Backpack LMs learn multiple sense vectors per word, allowing for explicit modeling of polysemous words. Previously tested in monolingual settings for English and Chinese, this project extends the Backpack architecture to multilingual modeling by training on both English and French using Europarl and MultiUN datasets. The multilingual Backpack LM efficiently encodes word meanings across languages, demonstrating lower perplexity and improved accuracy on cloze tasks compared to baseline GPT-2 models.

## Installation

<!--The project requires the following dependencies:
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
  -->
  
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

We save the checkpoints of the different models trained on Europarl and MultiUN to Google Drive.

|Model | Parameters| Number of sense vectors |
|:-----------------:|:--------------:|:-------------:|
|[Mini-GPT2](https://drive.google.com/file/d/1YxlRtqGeg-ISILtxDl0p6t4IrQR2qe-Y/view?usp=sharing)| 14M | - |
|[Mini-Backpack-16](https://drive.google.com/file/d/1Q3ZXjrMXZylwCGqyFoHfBzX2gf09z_M3/view?usp=sharing)| 19M | 16 |
|[Small-GPT2](https://drive.google.com/file/d/1gwbNGrDZ1MMR1L_nxfoQ1x9y_BgF5-gn/view?usp=sharing)| 93M | - |
|[Small-Backpack-16](https://drive.google.com/file/d/1bSEPVB42utEsIRyELnIgLQ0S9F0iyrIg/view?usp=sharing)| 112M | 16|


To train a Backpack LM model or GPT2, follow these steps:
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


## Key Findings
- Multilingual Training: This research marks the first application of Backpack LMs in multilingual settings, specifically training them on English and French corpora simultaneously.
- Efficient Learning: The models efficiently learn word meanings without encoding language-specific sense vectors, allowing them to handle polysemous words effectively.
- Performance Metrics: The Backpack LM (112M parameters) achieved lower perplexity scores compared to a baseline GPT2 (93M parameters). It slightly outperformed the baseline in a cloze task in top-1 accuracy.<!--, demonstrating superior context-dependent generation capabilities.-->
- Sense Vector Analysis: The study found that the sense distributions learned by the Backpack LMs do not vary significantly across languages, suggesting that these models can effectively share sense vectors between languages without losing semantic accuracy.


## Acknowledgements
This implementation is based on the papers [Backpack Language Models]() and [Character-level Chinese Backpack Language Models](https://arxiv.org/abs/2310.12751).

## References

- **Backpack Language Models**: [Backpack Language Models](https://arxiv.org/abs/2305.16765) by John Hewitt, John Thickstun, Christopher D. Manning, and Percy Liang (2023).
  
- **Character-level Chinese Backpack Language Models**: [Character-level Chinese Backpack Language Models](https://arxiv.org/abs/2310.12751) by Hao Sun and John Hewitt (2023).

