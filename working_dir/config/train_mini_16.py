
#@title config/train_mini_16.py


# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-gpt2-mini-16'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

meta_vocab_size = 10000

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'bkp'
wandb_run_name = 'bkp-mini-16'

dataset = ['europarl', 'multiun']
batch_size = 16
gradient_accumulation_steps = 64
block_size = 256 # context of up to 256 previous characters
xlm_alpha=1.0


# # baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0



# gpt2
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# bias = False # do we use bias inside LayerNorm and Linear layers?
n_sense_vector = 16


learning_rate = 6e-5 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 6e-6# 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 6000 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only

compile = True # do not torch compile the model
dtype = "float16"
