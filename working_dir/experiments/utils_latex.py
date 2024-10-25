#@title Utils for latex

# import numpy as np

def create_tab(word, tab_senses):

  # tab_senses = list(np.array(tab_senses).T)# transpose the content of the list
  n_col = len(tab_senses[0])
  n_row = len(tab_senses)

  tab_content = "\\begin{tabular}{c*{" + f"{n_col-1}" + "}{c}{c}}" + "\n\\hline\n"
  tab_content += "Sense &\\multicolumn{" + f"{n_col}" +"}{c}{" + f"{word}" + "}" + "\\\\\hline\n"

  for i in range(n_row):
    tab_content += f"{i+1} & " + " & ".join(tab_senses[i]) + "\\\\" + "\n"
  tab_content += "\\hline"
  tab_content += "\n\\end{tabular}"
  return tab_content

def print_senses(list_of_words, func_senses_1, func_senses_2):

  tabular = """\\documentclass[a4paper, 12pt, landscape]{article}
  \\usepackage{fourier}
  \\usepackage{tabulary}
  \\usepackage[margin=0.8cm]{geometry}
  \\usepackage{longtable}
  \\begin{document}
  \\begin{center}"""

  tabular += "\\begin{longtable}{*{" + f"{1}" + "}{c}}" + "\n"
  for i in range(1, len(list_of_words)+1):
    #get word
    word = list_of_words[i-1]
    #get senses
    try:
      tab_senses = func_senses_1(word)
    except:
      continue
      tab_senses = func_senses_2(word)
      word += " (new)"
    #create tab and add to tabular
    tabular += create_tab(word, tab_senses)
    # if i %2 == 1:
    #   tabular += "\n & \n"
    # else:
    #   tabular += "\\\\\\\\\n"
    tabular += "\\\\\\\\\n"

  tabular += "\\end{longtable}"

  tabular += "\n\\end{center}"
  tabular += "\n\\end{document}"

  with open('experiments/table_senses.tex', 'w') as f:
    f.write(tabular)
  print("output in experiments/table_senses.tex ...")
