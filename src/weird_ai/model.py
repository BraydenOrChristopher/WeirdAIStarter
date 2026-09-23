import torch
import torch.nn as nn

from weird_ai.transformer import TransformerBlock
from weird_ai.layer_norm import LayerNorm

class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, num_heads, num_layers, dropout=0.0, qkv_bias=False):
        super().__init__()

        # TODO:
        # Create embeddings
        # Create attention layers
        # Create output head

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(emb_dim, context_length, num_heads, dropout, qkv_bias)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim, vocab_size, bias=False)


    def forward(self, x):
        # TODO:
        # Implement forward pass
        
        batch_size, seq_len = x.shape

        token_embeds = self.token_embedding(x)
        position_ids = torch.arange(seq_len, device=x.device)
        position_embeds = self.position_embedding(position_ids)

        x = token_embeds + position_embeds
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits