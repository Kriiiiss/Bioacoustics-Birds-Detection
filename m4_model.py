"""M4: CNN-LSTM over the 4 clips [t-3, t-2, t-1, t]; only clip t is predicted.

ClipEncoder is the clip-level CNN (four conv blocks, mean over frequency, then max+mean pooling over time),
shared across the 4 clips and run only on real (unmasked) clips so padding does not enter BatchNorm. The 4
clip embeddings go through an LSTM cell; a padded step keeps the previous state (it is skipped). The final
state feeds two heads: known-species logits and one Unknown logit (independent sigmoids).

With context disabled the three earlier clips are masked, so the model sees a single clip: the matching
"no history" control under the identical pipeline.
"""
import torch
import torch.nn as nn

from model import conv_block


class ClipEncoder(nn.Module):
    """Clip-level CNN: (N, 1, 128, 500) log-mel -> (N, 256) embedding."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16, 1), conv_block(16, 32), conv_block(32, 128), conv_block(128, 256))

    def forward(self, x):                           # x: (N, 1, 128, 500)
        h = self.features(x).mean(2)                # mean over frequency -> (N, 256, T)
        return h.max(2).values + h.mean(2)          # max + mean over time -> (N, 256)


class CNNLSTM(nn.Module):
    def __init__(self, n_known, hidden=256, dropout=0.3, embed_dim=256):
        super().__init__()
        self.encoder = ClipEncoder()
        self.cell = nn.LSTMCell(embed_dim, hidden)
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.known_head = nn.Linear(hidden, n_known)
        self.unknown_head = nn.Linear(hidden, 1)

    def embed_clips(self, x, mask):
        """x (B, L, 128, 500), mask (B, L) -> (B, L, embed_dim); zeros for padded slots."""
        B, L = mask.shape
        e = x.new_zeros(B, L, self.embed_dim)
        e[mask] = self.encoder(x[mask].unsqueeze(1))
        return e

    def forward(self, x, mask):
        e = self.embed_clips(x, mask)
        h = e.new_zeros(e.shape[0], self.hidden)
        c = torch.zeros_like(h)
        for t in range(mask.shape[1]):
            h_new, c_new = self.cell(e[:, t], (h, c))
            keep = mask[:, t, None]
            h, c = torch.where(keep, h_new, h), torch.where(keep, c_new, c)
        h = self.dropout(h)
        return self.known_head(h), self.unknown_head(h)
