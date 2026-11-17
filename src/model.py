import torch
from torch import nn

class MiniParT(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project each jet's features into an embedding space.
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer encoder learns relationships between the two jets.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classify the combined event representation.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)

        # Combine the representations of the two jets.
        x = x.mean(dim=1)

        return self.mlp(x)
