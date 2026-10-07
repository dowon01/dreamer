import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

class Actor(nn.Module):
    # 평균은 tanh, std는 [min_std, max_std] 범위의 정규분포. 샘플은 [-1, 1]로 clip
    def __init__(self, latent_dim, action_dim=3, hidden_dim=512, min_std=0.1, max_std=1.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def get_dist(self, latent):
        x = self.net(latent)
        mu = torch.tanh(self.mu_head(x))
        std = (self.max_std - self.min_std) * torch.sigmoid(self.std_head(x) + 2.0) + self.min_std
        return Independent(Normal(mu, std), 1)

    def forward(self, latent, deterministic=False):
        dist = self.get_dist(latent)
        if deterministic:
            return dist.base_dist.loc
        return dist.sample().clamp(-1.0, 1.0)

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent):
        pred = self.net(latent)
        dist = Normal(pred, 1.0)
        dist = Independent(dist, 1)
        return dist
