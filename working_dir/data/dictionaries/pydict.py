
from PyMultiDictionary import MultiDictionary

dictionary = MultiDictionary()

def classify_subword(word):
  if dictionary.meaning('fr', word)[1]!= '':
    return "French"
  elif dictionary.meaning('en', word)[1]!='':
    return "English"
  else:
    return False
