from torch import nn as nn

#Other classes
from weird_ai.layer_norm import LayerNorm
from weird_ai.feed_forward import FeedForward
from weird_ai.attention import SelfAttention

class TransformerBlock(nn.Module):
    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        #  - LayerNorm
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        #  - SelfAttention from previous assignment
        self.attention = SelfAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            qkv_bias=qkv_bias
        )

        #  - FeedForward
        self.feed_forward = FeedForward(emb_dim=emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    #  - Residual connections
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
