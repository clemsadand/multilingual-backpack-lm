#@title Sense vector experiment
import torch
from loader import load_model, device
from transformers import GPT2LMHeadModel
import operator
from functools import reduce
from utils import TopK
from tabulate import tabulate
# from data.dictionaries.create_dict import classify_subword, is_english_word, is_french_word
from data.dictionaries.pydict import classify_subword
from experiments.utils_latex import create_tab, print_senses

exec(open('configurator.py').read())  # overrides from command line or config file

bkp_tokenizer= True


# @torch.no_grad()
class SenseVectorExperiment(object):
  def __init__(self):
    self.model, self.encode, self.decode = load_model("backpack-lm")
    self.sense_vector = self.model.sense_vectors(device=device)
    self.vocab_size = self.sense_vector.shape[0]
    self.word_vectors = self.model.wte(torch.arange(self.vocab_size).to(device))
    self.n_sense_vectors = self.sense_vector.shape[1]
    self.id2word = {k: self.decode([k]) for k in range(self.vocab_size)}
    # self.id2word = {}
    # for k in range(self.vocab_size):
    #   try:
    #     self.id2word[k] = self.decode([k])
    #   except:
    #     self.id2word[k] = "<UNK>"
    #     # print(k)

    # try:
    #   self.word2id = {word: self.encode(word)[1] for word in self.id2word.values()}
    # except IndexError:
    #   self.word2id = {word: self.encode(word)[0] for word in self.id2word.values()}
    self.word2id = {word: self.encode(word)[0] if len(self.encode(word)) > 0 else -1 for word in self.id2word.values()}

  @torch.no_grad()
  def sense_projection(self, word, k=5):
    senses = self.sense_vector[self.word2id[word]].to(device)
    output = self.model.backpack.logit_layer(senses)
    topk = torch.topk(output, k, dim=-1).indices.to('cpu').numpy()
    return [[self.id2word[i] for i in row] for row in topk]

  def sense_projection_new(self, word, k=5):
    #tokenize the word
    subwords = [self.decode([k]) for k in self.encode(word)]
    #get sense of each subword and average the senses
    # senses = self.sense_vector[self.word2id[word]].to(device)
    # output = self.model.backpack.logit_layer(senses)
    output = self.compositional_sense_projection(word, strategy='avg')
    topk = torch.topk(output, k, dim=-1).indices.to('cpu').numpy()
    return [[self.id2word[i] for i in row] for row in topk]

  def full_sense_projection(self, word, k=5):
    senses = self.sense_vector[self.word2id[word]].to(device)
    output = self.model.backpack.logit_layer(senses)
    topk = torch.sort(output, dim=-1, descending=True).indices.to('cpu').numpy()
    # filter = [[self.id2word[i] for i in row if classify_word(self.id2word[i]) ] for row in topk]
    filter = [[] for _ in range(self.n_sense_vectors)]
    for n in range(self.n_sense_vectors):
      n_iter = 0
      row = topk[n]
      for i in row:
        if classify_subword(self.id2word[i]):
          n_iter += 1
          filter[n].append(self.id2word[i])
        if n_iter == k:
          break
    return filter

  def full_sense_projection_new(self, word, k=5):
    #tokenize the word
    subwords = [self.decode([k]) for k in self.encode(word)]
    #get sense of each subword and average the senses
    # senses = self.sense_vector[self.word2id[word]].to(device)
    # output = self.model.backpack.logit_layer(senses)
    output = self.compositional_sense_projection(word, strategy='avg')
    topk = torch.sort(output, dim=-1, descending=True).indices.to('cpu').numpy()
    # filter = [[self.id2word[i] for i in row if classify_word(self.id2word[i]) ] for row in topk]
    filter = [[] for _ in range(self.n_sense_vectors)]
    for n in range(self.n_sense_vectors):
      n_iter = 0
      row = topk[n]
      for i in row:
        if classify_subword(self.id2word[i]):
          n_iter += 1
          filter[n].append(self.id2word[i])
        if n_iter == k:
          break
    return filter

  def visual_sen(self, word):
    print("Senses of word:", word)
    try:
      se = self.full_sense_projection(word)
    except:
      print(f"{word} not in vocabulary...")
      se = self.full_sense_projection_new(word)

    print(tabulate(se, headers='firstrow', tablefmt='fancy_grid'))


  def full_visual_sen(self, word):
    print("Senses of word:", word)
    try:
      se = self.full_sense_projection(word)
    except:
      print(f"{word} not in vocabulary...")
      se = self.full_sense_projection_new(word)

    print(tabulate(se, headers='firstrow', tablefmt='fancy_grid'))


  @torch.no_grad()
  def compositional_sense_projection(self, words, strategy='avg'):
    if strategy == 'avg':
      sense = reduce(operator.add, (self.sense_vector[self.word2id[word]] for word in words)) / len(words)
    elif strategy == 'contextualized':
      sense = self.get_contextualized_sense(words)
    else:
      raise NotImplementedError
    output = self.model.backpack.logit_layer(sense)
    return output

  @torch.no_grad()
  def cosine_similarity(self, x1id, x2id, device=device):
    return torch.cosine_similarity(self.sense_vector[x1id], self.sense_vector[x2id], dim=-1).to(device)

  @torch.no_grad()
  def min_sense_cosine(self, x1id, x2id):
    return torch.min(self.cosine_similarity(x1id, x2id))

  @torch.no_grad()
  def min_sense_cosine_matrix(self, words, k=5):
    sense_dict = {word: [TopK(k) for _ in range(self.n_sense_vectors)] for word in words}
    for word in words:
        print(f'analysing on word {word}')
        similarities = self.cosine_similarity(self.word2id[word], torch.arange(0, self.vocab_size))  # len, k
        for j in range(0, self.vocab_size):
            if j == self.word2id[word]:
                continue
            # l: index, v: score of similarities[j]
            for l, v in enumerate(similarities[j]):
                sense_dict[word][l].append((float(v), self.id2word[j]))
    return sense_dict

  def get_contextualized_sense(self, words):
    contextualized_sense = self.model.backpack.analyse_sense_contextualization(
        torch.tensor(self.encode(words)[1:- 1], dtype=torch.long, device=device).unsqueeze(0)
    )
    new_sense = contextualized_sense[-1, -1, :, :]
    return new_sense

  def semantic_sim_muse(self, sim_path, split_char=" "):
    file = open(sim_path)
    # word_sim_pre = {i: [] for i in range(self.model.config.n_sense_vector)}
    tab_cos_sim = torch.zeros((self.model.config.n_sense_vector, self.model.config.n_sense_vector)).to(device)
    n_examples = 0
    for line in file:
      word1, word2 = line.strip().split(split_char)
      #word1
      sense1 = reduce(operator.add, (self.sense_vector[k].to(device) for k in self.encode(word1))) / len(word1)
      sense2 = reduce(operator.add, (self.sense_vector[k].to(device) for k in self.encode(word2))) / len(word2)

      # print(sense2.shape, sense1.shape1)
      # cos_sim = torch.sum(sense1 * sense2, -1).detach().numpy()
      # cos_sim = torch.cosine_similarity(sense1, sense2, dim=-1).to(device)
      for i in range(self.model.config.n_sense_vector):
        for j in range(self.model.config.n_sense_vector):
          tab_cos_sim[i, j] += torch.cosine_similarity(sense1[i], sense2[j], dim=-1).to(device)
      # print(tab_cos_sim)
      # for i, sim in enumerate(cos_sim):
      #   word_sim_pre[i].append(sim)

      n_examples += 1
      # if n_examples > 200:
      #   break
    tab_cos_sim /= n_examples + 1
    print("Cosine similarity")
    print(tab_cos_sim)
    # for i in range(self.model.config.n_sense_vector):
    #   print(f'Sense:{i + 1} Score:{sum(word_sim_pre[i])/len(word_sim_pre[i])}')
    file.close()

if __name__ == "__main__":
  e = SenseVectorExperiment()
  # e.visual_sen("action")
  # e.full_visual_sen("action")
  # e.visual_sen("Europe")
  # e.visual_sen("gouvernement")
  # e.visual_sen("ministre")
  # e.visual_sen("energy")
  # e.visual_sen("énergie")
  # e.visual_sen("bank")
  # e.visual_sen("banque")
  # e.visual_sen("urbain")
  # e.visual_sen("urban")
  # # e.visual_sen("communication")
  #
  sim_path = "data/similarity/freq_words.txt"
  e.semantic_sim_muse(sim_path, split_char=",")

  # list_of_words1 = ["public", "Commission", "Europe", "union", "point"]#, 
  # list_of_words2 = ["Presidency", "Présidence", "Parliament", "Parlement"]#
  # list_of_words3 = [ "cooperation", "coopération", "Community", "Communauté", "States", "États"]

  print_senses(list_of_words3, e.sense_projection, e.sense_projection_new)
