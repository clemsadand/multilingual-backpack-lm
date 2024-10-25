
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

out_dir = "out-small-16"
exec(open('configurator.py').read())  # overrides from command line or config file

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bkp_tokenizer= True

# @torch.no_grad()
class SenseVectorExperiment(object):
  def __init__(self):
    self.model, self.encode, self.decode = load_model("backpack-lm")
    self.sense_vector = self.model.sense_vectors(device=device)
    self.vocab_size = self.sense_vector.shape[0]
    self.word_vectors = self.model.wte(torch.arange(self.vocab_size).to(device))
    self.n_sense_vectors = 8#self.sense_vector.shape[1]
    self.id2word = {k: self.decode([k]) for k in range(self.vocab_size)}
    self.word2id = {word: self.encode(word)[0] if len(self.encode(word)) > 0 else -1 for word in self.id2word.values()}

  @torch.no_grad()
  def sense_projection(self, word, k=5):
    senses = self.sense_vector[self.word2id[word]].to(device)
    output = self.model.backpack.logit_layer(senses)
    topk = torch.topk(output, k, dim=-1).indices.to('cpu').numpy()
    return [[self.id2word[i] for i in row] for row in topk]

  @torch.no_grad()
  def sense_projection_new(self, word, k=5):
    #tokenize the word
    subwords = [self.decode([k]) for k in self.encode(word)]
    #get sense of each subword and average the senses
    # senses = self.sense_vector[self.word2id[word]].to(device)
    # output = self.model.backpack.logit_layer(senses)
    output = self.compositional_sense_projection(word, strategy= 'avg')
    topk = torch.topk(output, k, dim=-1).indices.to('cpu').numpy()
    return [[self.id2word[i] for i in row] for row in topk]


  @torch.no_grad()
  def full_sense_projection(self, word, k=5):
    senses = self.sense_vector[self.word2id[word]].to(device)
    output = self.model.backpack.logit_layer(senses)
    # topk = torch.sort(output, dim=-1, descending=True).indices.to('cpu').numpy()
    topk = torch.topk(output, 100, dim=-1).indices.to('cpu').numpy()
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
  def get_contextualized_sense(self, words):
    contextualized_sense = self.model.backpack.analyse_sense_contextualization(
        torch.tensor(self.encode(words)[1:- 1], dtype=torch.long, device=device).unsqueeze(0)
    )
    new_sense = contextualized_sense[-1, -1, :, :]
    return new_sense

if  __name__ == "__main__":
  experiment = SenseVectorExperiment()
  # print(experiment.sense_projection(" action"))

  list_of_words = [
    "action", "opinion", "plan", "service", "communication", "motion",
    "government", "gouvernement", "bank", "urban", "economy", "technology",
    "Commission", "parliament", "states", "union", "report", "council",
    "people", "political", "believe", "situation", "point", "président",
    "parlement", "state", "crois", "commissaire", "citizens", "citoyens",
    "Europe", "Parliament", "Parlement", "President", "Président", "members",
    "membres", "Council", "Conseil", "directive", "vote", "democracy",
    "démocratie", "Union", "policy", "politique", "legislation", "législation",
    "trade", "commerce", "agriculture", "budget", "economic", "économique",
    "development", "développement", "rights", "droits", "justice",
    "environment", "environnement", "protection", "treaty", "traité", "States",
    "agreement", "accord", "cooperation", "coopération", "security",
    "sécurité", "public", "services", "health", "santé", "innovation",
    "social", "market", "marché", "competition", "concurrence", "reform",
    "réforme", "freedom", "liberté", "migration", "workers", "travailleurs",
    "infrastructure", "change", "changement", "targets", "objectifs",
    "transparency", "transparence", "sessions", "budgetary", "budgétaire",
    "borders", "frontières", "negotiations", "négociations", "central",
    "fiscal", "regions", "régions", "transport", "accountability",
    "responsabilité", "solidarity", "solidarité", "the", "le", "is", "est",
    "in", "dans", "it", "il", "at", "à", "by", "par", "we", "nous", "they",
    "ils", "you", "vous", "she", "elle", "he", "with", "avec", "as", "comme",
    "on", "sur", "for", "pour", "from", "de", "this", "ce", "that", "cela",
    "these", "ces", "those", "ceux", "but", "mais", "or", "ou", "and", "et",
    "if", "si", "since", "depuis", "after", "après", "before", "avant",
    "during", "pendant", "when", "quand", "now", "maintenant", "then",
    "alors", "some", "quelques", "all", "tout", "no", "non", "not", "pas",
    "never", "jamais", "always", "toujours", "often", "souvent", "here",
    "ici", "there", "là", "where", "où", "how", "comment", "who", "qui",
    "why", "pourquoi", "very", "très", "much", "beaucoup", "few", "peu",
    "more", "plus", "less", "moins", "good", "bon", "better", "mieux",
    "new", "nouveau", "first", "premier", "last", "dernier", "same", "même",
    "different", "différent", "big", "grand", "small", "petit", "long",
    "short", "court", "low", "bas", "strong", "fort", "hard", "dur",
    "difficult", "difficile", "quick", "rapide", "right", "droit",
    "important", "nécessaire", "possible", "impossible", "European",
    "amendement", "law", "Member", "emploi", "industrie", "énergie",
    "égalité", "attention", "commitment", "engagement", "comprehensive",
    "global", "countries", "pays", "data", "données", "decisions",
    "décisions", "discussion", "distribution", "efforts", "essential",
    "essentiel", "framework", "cadre", "growth", "croissance", "impact",
    "information", "international", "leadership", "legal", "juridique",
    "management", "gestion", "national", "obligations", "organizations",
    "organisations", "outcomes", "résultats", "participation", "peace", "paix",
    "personnes", "population", "principles", "principes", "relations",
    "rapport", "resolution", "résolution", "resources", "ressources",
    "respect", "risk", "role", "rôle", "vision", "women", "femmes", "violence"
  ]



  # list_of_words = ["democracy", "droits", "démocratie", "emploi", "first", "good", "how", "justice", "law", "moins", "nouveau", "nécessaire", "où", "politique", "quick", "rights", "rights", "égalité"]

  list_of_words = [" "+word for word in list_of_words]
  print_senses(list_of_words, experiment.sense_projection,None)


  # Utils latex Sense
  # list_of_words = ["rights", "law", "quick"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 4, experiment.sense_projection,None, file_path="sense4_en.tex")

  # list_of_words = ["égalité", "emploi", "nécessaire"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 4, experiment.sense_projection,None, file_path="sense4_fr.tex")

  # list_of_words = ["democracy", "justice", "rights"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 1, experiment.sense_projection,None, file_path="sense1_en.tex")

  # list_of_words = ["droits", "démocratie", "politique"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 1, experiment.sense_projection,None, file_path="sense1_fr.tex")

  # list_of_words = ["first", "good", "how"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 7, experiment.sense_projection,None, file_path="sense7_en.tex")

  # list_of_words = ["moins", "nouveau", "où"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, 7, experiment.sense_projection,None, file_path="sense7_fr.tex")


  # Utils latex 2
  # list_of_words = ["democracy", "economy", "education", "problem", "technology"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="cognates_en.tex")

  # list_of_words = ["démocratie", "économie", "éducation", "problème", "technologie"]
  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="cognates_fr.tex")

  # list_of_words = ["countries", "need", "new", "policy", "rights", "work"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="interesting_en.tex")

  # list_of_words = ["pays", "besoin", "nouveau", "politique", "droits", "travail"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="interesting_fr.tex")

  # list_of_words = ["respect", "social", "violence", "nation", "information", "population"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="shared_words.tex")

  # list_of_words = ["après", "comment", "pendant", "citoyens", "frontières", "gouvernement", "responsabilité"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="french.tex")

  # list_of_words = ["after", "how", "during", "borders", "government", "accountability"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="english.tex")

  # list_of_words = ["action", "innovation", "public", "respect", "social", "vote"]

  # list_of_words = [" "+word for word in list_of_words]
  # print_senses(list_of_words, experiment.sense_projection,None, file_path="shared.tex")




