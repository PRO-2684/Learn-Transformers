from torch import nn, cat
from scaled_dot_product_attention import scaled_dot_product_attention

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        return scaled_dot_product_attention(Q, K, V, query_mask, key_mask, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        heads = [head(query, key, value, query_mask, key_mask, mask) for head in self.heads]
        return self.linear(cat(heads, dim=-1))

if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    config = AutoConfig.from_pretrained(model_ckpt)
    token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    embeddings = token_embeddings(inputs.input_ids)

    multi_head = MultiHeadAttention(config)
    query = key = value = embeddings
    attention = multi_head(query, key, value)
    print("Multi-head attention:", attention.shape, attention)
