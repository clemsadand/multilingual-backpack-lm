#@title Cloze task vf
import os
import json
import torch
from torch.nn import functional as F
from loader import load_model


out_dir = "out-small-16"
model_name = "backpack-lm"
device = "cuda" if torch.cuda.is_available() else "cpu"
exec(open('configurator.py').read())  # overrides from command line or config file


def load_dev_set(dev_set_path):
    with open(dev_set_path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]


def cloze_test(paths, model_name=model_name, k=3):

    model, encode, decode = load_model(model_name=model_name)
    model = model.to(device)

    for path in paths:
      top1, top3 = 0, 0
      dev_set = load_dev_set(path)

      for data in dev_set:
          cloze_passage = data['cloze_passage']
          true_word = data['answer']
          max_length = data['length_of_sentences']

          start = cloze_passage[:cloze_passage.index(' <mask>')]
          start_tokens = encode(start)[:- 1]

          beam_search = [(torch.tensor(start_tokens, dtype=torch.long, device=device), 1)]
          new_beam = []

          while len(beam_search[0][0]) < max_length:
              new_beam.clear()
              for tokens, prob in beam_search:
                  logits = model(tokens[None, ...].to(device))[0][0, -1, :]
                  topk = torch.topk(F.softmax(logits, dim=-1), k=k + 2)

                  for index, new_prob in zip(topk.indices, topk.values):
                      if decode(index) not in [
                          ',', '，', '。', '[UNK]', '！', '!', '；', ';', '?', '？', '[ U N K ]', '"', '”', '“', '、',
                          '：', ':', '「', '」', '【', '】', '`', '…', '……'
                      ] and index != 100:
                          new_tokens = torch.cat((tokens, torch.tensor([index]).to(device)))
                          new_beam.append((new_tokens, new_prob * prob))

              beam_search = sorted(new_beam, key=lambda x: -x[1])[:k]

              if len(beam_search[0][0]) >= max_length:
                  break

          for i, (tokens, _) in enumerate(beam_search):
              word = decode(tokens[len(start_tokens):])#.replace(' ', '')
              if true_word in word:
                  if i == 0:
                      top1 += 1
                  top3 += 1
                  break

      print(f"Evaluating with {path}")
      print(f"Top-1 Accuracy: {top1 / len(dev_set):0.4f}")
      print(f"Top-3 Accuracy: {top3 / len(dev_set):0.4f}")
      print("-"*10)


if __name__ == '__main__':
  # paths = ["data/cloze_data/eu.en.jsonl", "data/cloze_data/eu.fr.jsonl", "data/cloze_data/un.en.jsonl", "data/cloze_data/un.fr.jsonl"]
  cloze_data_dir = "data/cloze_data"
  paths = [os.path.join(cloze_data_dir, fname) for fname in os.listdir(cloze_data_dir) if fname.split(".")[-1] == "jsonl"]
  cloze_test(paths, k=3)
