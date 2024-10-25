#@title Perplexity per language
import os
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import torch

from loader import load_model, get_batch, device_type
from contextlib import nullcontext
# from config.train_europarl import batch_size, block_size, eval_iters

###########################################################
batch_size = 8#same as in the train_europarl.py (same as the train configuration)
block_size = 256#same as in the train_europarl.py
eval_iters = 100
# ###################
# # evaluation_data_path = 'data/europarl/en_evaluation.bin'
evaluation_data_dirs = 'data/europarl,data/multiun'
device_type = 'cuda'
dtype = 'float16'#'bfloat16'
data_bin_dtype = 'uint16'
model_name = "backpackl-lm" #"backpackl-lm" # "gpt2"
out_dir = "out-"
# data_bin_dtype = 'uint32'  # force overwrite
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# *********************************************************
exec(open('configurator.py').read())  # overrides from command line or config file
# *********************************************************

print(f"Evaluating {model_name}...")

def percentage_data_sample_func(data, block_size, batch_size):
    return torch.randint(int(len(data) * 0.6) - block_size, (batch_size,)) + int(len(data) * 0.4)


def evaluate_single_language(evaluation_data_path, model, is_percentage):
    eval_data = np.memmap(evaluation_data_path, dtype=getattr(np, data_bin_dtype), mode='r')
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(eval_data, sample_func=percentage_data_sample_func if is_percentage else None)
        with ctx:
            _, loss = model(X, Y)
        losses[k] = torch.exp(loss).item()
    return losses.mean()


@torch.no_grad()
def estimate_perplexity(model_name):
    model = load_model(model_name=model_name)[0]
    model.eval()
    evaluation_output = {}
    langs  = set()
    dir_i = 0
    for dir_ in evaluation_data_dirs.split(','):
        dir_i += 1
        for f_name in os.listdir(dir_):
            if f_name.endswith('_evaluation1.bin'):
                is_percentage = False
                lang_name = f_name.split('_evaluation1')[0]
                langs.add(lang_name)
                print(f"Found {f_name} in {dir_}...")
            elif f_name.endswith('_evaluation2.bin'):
                is_percentage = True
                lang_name = f_name.split('_evaluation2')[0]
                langs.add(lang_name)
                print(f"Found {f_name} in {dir_}...")
            # elif os.path.isdir(os.path.join(dir_, f_name)):
            #     is_percentage = True
            #     lang_name = f_name
            #     f_name = os.path.join(f_name, f'{f_name}_99.bin')
            else:
                continue
            perp = evaluate_single_language(os.path.join(dir_, f_name), model, is_percentage)
            print('perplexity for {} is {}'.format(lang_name, perp))

            if not (lang_name in evaluation_output):
              evaluation_output[lang_name] = perp
            else:
              evaluation_output[lang_name] = evaluation_output[lang_name] + perp
    return [{lang:evaluation_output[lang]/dir_i} for lang in langs]


if __name__ == "__main__":
    print(estimate_perplexity(model_name))
