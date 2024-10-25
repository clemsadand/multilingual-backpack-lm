
#@title Tokenization

import os
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import GPT2TokenizerFast
import re

# Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/BkpTokenizer.json")

# Read data from the text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts

# Process text into tokens
def process(example):
    ids = tokenizer.encode(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out

# Prepare the data and combine the results into the same binary files
def prepare(list_of_data_paths, dir_i=2):
    """Process all files in list_of_data_paths and append results to the same binary files for each split"""
    print("Preparing data...")

    num_of_tokens = 0
    num_of_tokens_train = 0
    file_count = 0

    for data_path in list_of_data_paths:
        for f_name in os.listdir(data_path):
            file_count += 1
            lang = ''
            extension = ''
            if f_name.endswith(".txt"):
                lang = f_name.split(".txt")[0]
                extension = "txt"
            elif re.search(r"\.txt\.\d{1,2}$", f_name):
                lang = f_name.split(".txt.")[0]
                extension = ".".join(f_name.split(".")[1:])
            else:
                continue
            f_path = os.path.join(data_path, f"{lang}.{extension}")
            print(f"Processing {f_path}...")

            # Load data
            texts = read_text_file(f_path)

            # Create Dataset from loaded text data
            dataset = Dataset.from_dict({'text': texts})

            # Split the dataset into train, validation, and evaluation sets
            dataset = dataset.train_test_split(test_size=0.01, seed=2357, shuffle=dataset.num_rows < 10000000)
            train_set = dataset['train']
            split_dataset = dataset['test'].train_test_split(test_size=0.5, seed=2357, shuffle=True)
            split_dataset['val'] = split_dataset.pop('test')
            split_dataset['evaluation'] = split_dataset.pop('train')
            split_dataset['train'] = train_set

            f_path = os.path.join(data_path, f"{lang}.{extension}")
            print(f_path+"...")
            #Load data
            texts = read_text_file(f_path)

            #Create a Dataset from the loaded text data
            dataset = Dataset.from_dict({'text': texts})
            print("Dataset created...")

            # Tokenize the dataset
            tokenized = split_dataset.map(
                process,
                remove_columns=["text"],
                desc="Tokenizing the splits",
                num_proc=os.cpu_count()
            )


            #Number of tokens used in this dataset
            print(f"Length of train split: {len(tokenized['train'])}")
            print(f"Length of val split: {len(tokenized['val'])}")
            print(f"Length of evaluation split: {len(tokenized['evaluation'])}")

            num_of_tokens += len(tokenized['train']) + len(tokenized['val']) + len(tokenized['evaluation'])
            num_of_tokens_train += len(tokenized['train'])


            # Save the tokenized data into binary files
            for split, dset in tokenized.items():
                filename = os.path.join(os.path.dirname(__file__), f'{lang}_{split}{dir_i}.bin')

                # Calculate the new array length (total tokens so far)
                new_arr_len = np.sum(dset["len"])

                # Open existing binary file or create a new one in append mode
                if os.path.exists(filename):
                    # If the binary file exists, open it in read+write mode and append
                    arr = np.memmap(filename, dtype=np.uint16, mode='r+')
                    old_arr_len = arr.shape[0]
                    arr = np.memmap(filename, dtype=np.uint16, mode='r+', shape=(old_arr_len + new_arr_len,))
                else:
                    # If the file doesn't exist, create a new one
                    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(new_arr_len,))
                    old_arr_len = 0

                print(f"Appending to {filename}...")

                # Write new tokenized data to the file
                idx = old_arr_len  # start at the previous length (for appending)
                for example in tqdm(dset):
                    arr[idx: idx + example['len']] = example['ids']
                    idx += example['len']
                arr.flush()

            # Remove the processed file
            os.remove(f_path)

    print(f"Processed {file_count} files.")
    print(f"Number of tokens processed: {num_of_tokens}")
    print(f"Number of tokens in train set: {num_of_tokens_train}")

if __name__ == "__main__":
    list_of_data_paths = ["data/europarl/"]
    prepare(list_of_data_paths)
