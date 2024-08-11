from torch import nn, bmm
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer
from math import sqrt

MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL) # Load tokenizer

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False) # Tokenized input
print("Input IDs:", inputs.input_ids)

config = AutoConfig.from_pretrained(MODEL)
token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) # Token embeddings
print("Token embedding:", token_embeddings)

embeddings = token_embeddings(inputs.input_ids) # Embedding lookup
print("Embedding size:", embeddings.size())

Q = K = V = embeddings # Query, Key, and Value
dim_k = K.size(-1) # Dimension of Key
scores = bmm(Q, K.transpose(1, 2)) / sqrt(dim_k) # Scaled dot-product attention
print("Attention scores:", scores)

weights = F.softmax(scores, dim=-1) # Attention weights
print("Attention weights:", weights, "Sum:", weights.sum(dim=-1))

attention = bmm(weights, V) # Attention output
print("Attention:", attention.shape, attention)
