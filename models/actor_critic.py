import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim=3, hidden_dim=512, mean_scale=5.0, min_std=1e-4):
        super().__init__()
        self.mean_scale = mean_scale
        self.min_std = min_std
        
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
        
        mu = self.mu_head(x)
        mu = mu / self.mean_scale
        mu = torch.tanh(mu)
        mu = mu * self.mean_scale
        
        std = self.std_head(x)
        std = F.softplus(std) + self.min_std
        
        dist = Normal(mu, std)
        dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
        dist = Independent(dist, 1)
        
        return dist

    def get_action_and_log_prob(self, latent):
        dist = self.get_dist(latent)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def forward(self, latent, deterministic=False):
        dist = self.get_dist(latent)
        if deterministic:
            return torch.tanh(dist.base_dist.base_dist.loc)
        else:
            return dist.rsample()

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