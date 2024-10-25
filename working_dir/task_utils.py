
#@title Helper Function

import os
import re
import numpy as np
import torch
# from transformers import BertTokenizer, AutoTokenizer


def get_batch_function_for_multilingual_training(list_of_data_dir, data_bin_dtype, xlm_alpha, block_size, batch_size, device):
    """Return a function which select a batch during training"""
    #Define the device
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    #get languages
    langs = list(set([
		re.split(r'_train\d+\.bin', n)[0] 
		for data_dir in list_of_data_dir 
		for n in os.listdir(data_dir) 
		if re.search(r'_train\d+\.bin$', n)  
	]))
	
    #Put all together
    full_lang_dic = {
        lang: {
            'train': [np.memmap(os.path.join(data_dir, file_name), dtype=getattr(np, data_bin_dtype), mode='r')
              for data_dir in list_of_data_dir
              for file_name in os.listdir(data_dir) if re.search(fr'{lang}_train\d+\.bin', file_name)],
            'val': [np.memmap(os.path.join(data_dir, file_name), dtype=getattr(np, data_bin_dtype), mode='r')
              for data_dir in list_of_data_dir
              for file_name in os.listdir(data_dir) if re.search(fr'{lang}_val\d+\.bin', file_name)]
        } for lang in langs
    }

    #Get size of each dataset
    n_dict = {}
    for lang, v in full_lang_dic.items():
        n_dict[lang] = len(v['train'])

    #Proportion of dataset per lang
    p_dict = {lang: n / sum(n_dict.values()) for lang, n in n_dict.items()}
    q_dict = {lang: p ** xlm_alpha / sum(pp ** xlm_alpha for pp in p_dict.values()) for lang, p in p_dict.items()}


    #Function to load batch from the file
    def get_batch(split):
        lang = np.random.choice(list(q_dict.keys()), p=list(q_dict.values()))

        if lang in full_lang_dic:
            idx = np.random.randint(len(full_lang_dic["en"]["val"]))
            data = full_lang_dic[lang]['train'][idx] if split == 'train' else full_lang_dic[lang]['val'][idx]
            max_pointer = len(data)

        #
        ix = torch.randint(max_pointer - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    return get_batch


if __name__ == "__main__":
    get_batch = get_batch_function_for_multilingual_training(data_dir=["data/europarl/"], data_bin_dtype="uint16", xlm_alpha=0.3, block_size=256, batch_size=2, device="cpu")
    for i in range(10):
        X, Y = get_batch("train")
        print(f"Shape X:{X.shape}, Y:{Y.shape}")
