#@title Tokenizer training
# thanks to https://huggingface.co/learn/nlp-course/en/chapter6/8
# Load the tokenizer with
# wrapped_tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/BkpTokenizer.json")
import os
from tokenizers import (
    decoders,
    models,
    # normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# *********************************************************
# # List of txt files containing the corpora
# corpora = ["data/europarl/en.txt", "data/europarl/fr.txt", "data/multiun/en.txt", "data/multiun/fr.txt"]
list_of_data_paths = ["data/europarl", "data/multiun"]

corpora = [
    os.path.join(data_path, f_name) for data_path in list_of_data_paths for f_name in os.listdir(data_path)
]

vocab_size = 10001 #50257#Vocabulary size, same GPT2 tokenizer
# **********************************************************


#Load GPT2-tokenizer
tokenizer = Tokenizer(models.BPE())
#pre-tokenizers
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)#don't add space at begining of sentences
#set special end-of-text tokens
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
# train the tokenizers from texts
# tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
#train from txt files
tokenizer.train(corpora, trainer=trainer)
#Whether to trim the whitespaces from the produced offsets
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
#add decoder
tokenizer.decoder = decoders.ByteLevel()
# ************************
#Make the tokenizer loadable from transformers
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
# ************************
#save the tokenizer
wrapped_tokenizer.save_pretrained("tokenizer/BkpTokenizer.json")
