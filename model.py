"""Candidate CNN preserving frequency position with per-class temporal attention.

This is a hypothesis, not a proven improvement. It uses no pretrained weights.
"""
import torch
from torch import nn


def conv_block(c_in, c_out, n_convs=2):
    layers = []
    for j in range(n_convs):
        layers.extend([nn.Conv2d(c_in if j == 0 else c_out, c_out, 3, padding=1),
                       nn.BatchNorm2d(c_out), nn.ReLU()])
    return nn.Sequential(*layers, nn.AvgPool2d(2))


class ImprovedBirdCNN(nn.Module):
    """Same front-end as SlimCNN, but frequency-aware class-specific pooling."""

    def __init__(self, n_known, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16, 1), conv_block(16, 32),
                                      conv_block(32, 128), conv_block(128, 256))
        # Input has 128 mel bands; after four factor-two pools there are 8.
        # Keep their order instead of averaging away frequency position.
        self.project = nn.Sequential(nn.Conv1d(256 * 8, 128, 1), nn.ReLU(), nn.Dropout(dropout))
        self.frame_scores = nn.Conv1d(128, n_known + 1, 1)
        self.attention = nn.Conv1d(128, n_known + 1, 1)
        self.n_known = n_known

    def forward(self, x):
        h = self.features(x)
        if h.shape[2] != 8:
            raise ValueError("expected input with 128 mel bands")
        h = self.project(h.flatten(1, 2))
        scores = self.frame_scores(h)
        weights = torch.softmax(self.attention(h), dim=-1)
        logits = (scores * weights).sum(dim=-1)
        return logits[:, :self.n_known], logits[:, self.n_known:]
