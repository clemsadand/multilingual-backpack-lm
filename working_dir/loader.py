#@title load model from out-europarl

import os
import pickle

import numpy as np
import torch
import tiktoken
from backpack import BackpackLM, BackpackLMConfig
from model import GPT, GPTConfig
from transformers import GPT2TokenizerFast

tokenizer_name = "backpack"  # "gpt2"

#######################################################
batch_size = 8#same as in the train_europarl.py (same as the train configuration)
block_size = 256#same as in the train_europarl.py
#######################################################
seed = 1337
model_name = "backpack-lm"
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
out_dir = 'out-europarl-nano'
compile = False
dtype = 'float16'  # 'float32' or 'bfloat16' or 'float16'

exec(open('configurator.py').read())  # overrides from command line or config file
#######################################################

def load_model(model_name = model_name, force_model=None, force_dir=None):
  # torch.manual_seed(seed)
  # torch.cuda.manual_seed(seed)
  # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
  # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
  # init from a model saved in a specific directory
  #Checkpoint
  ckpt_path = os.path.join(out_dir, "ckpt.pt")
  checkpoint = torch.load(ckpt_path, map_location=device)
  # *************************************************
  Model = BackpackLM if model_name == 'backpack-lm' else GPT
  Config = BackpackLMConfig if model_name == 'backpack-lm' else GPTConfig
  config = Config(**checkpoint["model_args"])
  # print(model_name == 'backpack-lm')
  # *************************************************
  # Load model
  model = Model(config)
  # model = (force_model if force_model else Model)(config)
  #
  state_dict = checkpoint["model"]
  unwanted_prefix = "_orig_mod."
  for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict)
  model.eval()
  model.to(device)

  #
  if compile:
    model = torch.compile(model)# requires PyTorch 2.0 (optional)

  # look for the meta pickle in case it is available in the dataset folder
  # load_meta = False
  # if 'config' in checkpoint and 'dataset' in checkpoint['config']:  # older checkpoints might not have these...
  #   meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
  #   load_meta = os.path.exists(meta_path)

  #
  if tokenizer_name == "backpack":
    print(f"Using {tokenizer_name} encodings...")
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/BkpTokenizer.json")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda l: tokenizer.decode(l, clean_up_tokenization_spaces=True)
  else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
  return model, encode, decode



####################################################3#
device_type = 'cuda' if 'cuda' in device else 'cpu'
def get_batch(data, sample_func=None):
  if sample_func:
      ix = sample_func(data, block_size, batch_size)
  else:
      ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
  if device_type == 'cuda':
      # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
      x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  else:
      x, y = x.to(device), y.to(device)
  return x, y

if __name__ == "__main__":
  model, encode, decode = load_model(model_name="backpack-lm")
  print(encode("text"))
