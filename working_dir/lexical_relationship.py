#@title Lexical Relationship

import torch
from loader import load_model, device
from transformers import GPT2LMHeadModel
import operator
from functools import reduce
from utils import TopK
from tabulate import tabulate
# from data.dictionaries.create_dict import classify_subword, is_english_word, is_french_word
# from data.dictionaries.pydict import classify_subword
# from experiments.utils_latex import create_tab, print_senses
import pandas as pd
import numpy as np

# device = "cpu"
model_name = "backpack-lm"

exec(open('configurator.py').read())  # overrides from command line or config file

bkp_tokenizer= True

# @torch.no_grad()
class SenseVectorExperiment3(object):

  def __init__(self):
    self.model, self.encode, self.decode = load_model(model_name)
    self.vocab_size = 50256#self.sense_vector.shape[0]
    if model_name=="backpack-lm":
      self.sense_vector = self.model.sense_vectors(device=device)
      self.n_sense_vectors = self.sense_vector.shape[1]
      self.word_vectors = self.model.wte(torch.arange(self.vocab_size).to(device))
    else:
      self.word_vectors = self.model.transformer.wte(torch.arange(self.vocab_size).to(device))

    self.id2word = {k: self.decode([k]) for k in range(self.vocab_size)}
    self.word2id = {word: self.encode(word)[0] if len(self.encode(word)) > 0 else -1 for word in self.id2word.values()}

  @torch.no_grad()
  def semantic_sim(self, pair_words, model_name):
    # file = open(sim_path)
    word_sim_std = []

    # tab_cos_sim = torch.zeros((self.model.config.n_sense_vector, self.model.config.n_sense_vector)).to(device)

    n_examples = 0
    #
    if model_name =="backpack-lm":
      d = torch.zeros(self.model.config.n_sense_vector).to(device)
      word_sim_pre = {i: [] for i in range(self.model.config.n_sense_vector)}
    elif model_name=="gpt2":
      d = 0
      word_sim_pre = []
    else:
      raise ValueError(f"{model_name} not defined...")


    # d2 = torch.zeros(self.model.config.n_sense_vector).to(device)
    #
    N = len(pair_words)
    #
    for i in range(N):
      word1 = pair_words.iloc[i, 0]
      word2 = pair_words.iloc[i, 1]
      score = pair_words.iloc[i, 2]
      word_sim_std.append(float(score))

      # print(w1, w2)
      # word1, word2 = line.strip().split(split_char)
      #word1
      emb_vec = self.sense_vector if model_name=="backpack-lm" else self.word_vectors
      sense1 = reduce(operator.add, (emb_vec[k].to(device) for k in self.encode(word1))) / len(word1)
      sense2 = reduce(operator.add, (emb_vec[k].to(device) for k in self.encode(word2))) / len(word2)

      # print(sense2.shape, sense1.shape1)
      # cos_sim = torch.sum(sense1 * sense2, -1).detach().numpy()
      cos_sim = torch.cosine_similarity(sense1, sense2, dim=-1).to(device)

      #
      d += (cos_sim - score)**2
      if model_name == "backpack-lm":
        for i, sim in enumerate(cos_sim):
          word_sim_pre[i].append(abs(sim))
      else:
        word_sim_pre.append(abs(cos_sim))

      n_examples += 1

    #
    print(' '*7 +'Pearson ' +' '*2+ ' Spearman')

    if model_name == "backpack-lm":
      for i in range(self.model.config.n_sense_vector):
        # Pearson correlation
        corr_coef = np.corrcoef(word_sim_std, word_sim_pre[i])[0, 1]
        #
        print(f'Sense{i+1}: {corr_coef:0.5f} {1 - ((6 * d[i])/(n_examples*(n_examples*n_examples - 1))):0.5f}')
    elif model_name=="gpt2":
      corr_coef = np.corrcoef(word_sim_std, word_sim_pre)[0, 1]
      print(f'Sense{i+1}: {corr_coef:0.5f} {1 - ((6 * d)/(n_examples*(n_examples*n_examples - 1))):0.5f}')
    else:
      raise ValueError(f"{model_name} not defined...")

if __name__ == "__main__":

  #Load MultiSimLex data
  df = pd.read_csv('data/similarity/translation.csv')
  multisimlex = df[["ENG 1", "FRA 1"]].copy()
  scores = pd.read_csv('data/similarity/scores.csv')
  multisimlex["score"] = scores["FRA"]/6

  #
  experiment = SenseVectorExperiment3()
  experiment.semantic_sim(multisimlex[:500], model_name)
