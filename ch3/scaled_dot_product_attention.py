from torch import nn, bmm
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer
from math import sqrt

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return bmm(weights, value)

if __name__ == "__main__":
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
