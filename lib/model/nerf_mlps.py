"""NeRF models.

Contains the various models and sub-models used to train a Neural Radiance Field (NeRF).
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer


class NeRFModel(nn.Module):
    def __init__(self, position_dim=10, direction_dim=4, hidden_dim=256):
        super(NeRFModel, self).__init__()
        self.position_dim = position_dim
        self.direction_dim = direction_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.feature_fn = nn.Sequential(
            nn.Linear(hidden_dim + self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.density_fn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(hidden_dim + self.direction_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, pos_enc_samples, pos_enc_ray_dir): 
        x_features = self.mlp(pos_enc_samples)
        x_features = torch.cat((x_features, pos_enc_samples), dim=-1)
        x_features = self.feature_fn(x_features)
        density = self.density_fn(x_features)
        dim_features = torch.cat((x_features, pos_enc_ray_dir), dim=-1)
        rgb = self.rgb_fn(dim_features)
        return rgb, density
