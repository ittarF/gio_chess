#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.position_embeddings


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention block
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        # Feedforward block
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))  # Add & Norm
        return x


class ChessEvaluatorTransformer(nn.Module):
    def __init__(
        self, embed_dim=64, num_heads=4, mlp_dim=128, num_blocks=4, dropout=0.1
    ):
        super(ChessEvaluatorTransformer, self).__init__()
        self.num_patches = 8 * 8  # Each square is a patch
        self.embed_dim = embed_dim

        # Patch embedding: Flatten the features of each patch and linearly project
        self.patch_embedding = nn.Linear(5, embed_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, self.num_patches)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Output head: Pooling and final linear layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape input to patches: (B, 8, 8, 5) -> (B, 64, 5)
        x = x.view(batch_size, self.num_patches, -1)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, 64, embed_dim)

        # Add positional encodings
        x = self.positional_encoding(x)

        # Transpose for Transformer: (B, 64, embed_dim) -> (64, B, embed_dim)
        x = x.transpose(0, 1)

        # Pass through Transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Pool the sequence output: (64, B, embed_dim) -> (B, embed_dim)
        x = x.mean(dim=0)  # Alternatively, use a [CLS] token if added

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output in [-1, 1]
        return x


if __name__ == "__main__":

    model = ChessEvaluatorTransformer()
    input_tensor = torch.randn((1, 8, 8, 5))  # Batch size 1, 8x8 board with 5 features
    output = model(input_tensor)
    print(output)
    print(sum(p.numel() for p in model.parameters()))
