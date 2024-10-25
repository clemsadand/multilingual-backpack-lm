
#@title Build English and French dictionaries

# Create two sets of vocabularies:
# one for French (fr_dict, 404836) and
# one for English (en_dict, 466549 words)

# ************************************************************
import os
import subprocess

if not os.path.exists("data/dictionaries/words.txt"):
  print("download English and French words...")
  os.chdir("data/dictionaries")
  # ref: https://github.com/dwyl/english-words/tree/master
  subprocess.run(["wget", "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"])

  # ref: https://github.com/hbenbel/French-Dictionary/tree/master
  subprocess.run(["wget", "https://raw.githubusercontent.com/hbenbel/French-Dictionary/master/dictionary/dictionary.csv"])
  # %cd /content/{main_path}
  # %mv data/dictionaries/cre

import pandas as pd


# Load words
# en
df = pd.read_csv("data/dictionaries/words.txt", header=None, names=["word"], sep='\t', on_bad_lines='skip')
en_dict = set(df.iloc[:,0])
#rm
# os.remove("words.txt")

# fr
df = pd.read_csv("data/dictionaries/dictionary.csv")
# get dict
fr_dict = set(df.iloc[:,0])
# rm
# os.remove("dictionary.csv")
# fr_dict.remove("-")
# **************************************************************

def is_english_word(word):
  return word in en_dict or word.lower() in en_dict

def is_french_word(word):
  return word in fr_dict or word.lower() in fr_dict

def existing_word(word):
  return True if is_french_word(word) or is_english_word(word) else False
