import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, Independent, Bernoulli

# ==========================================
# 1. 유틸리티 함수: STE(Straight-Through Estimator) 샘플링
# ==========================================
def get_categorical_state(logits, stoch_dim=64, discrete_dim=32):
    # 이산 잠재 변수(Categorical Latent)를 안정적으로 샘플링
    logits = torch.clamp(logits, -20.0, 20.0)
    
    shape = logits.shape
    logits_reshaped = torch.reshape(logits, [*shape[:-1], stoch_dim, discrete_dim])
    
    dist = OneHotCategorical(logits=logits_reshaped)
    
    # STE (Straight-Through Estimator)
    sample = dist.sample()
    stoch = sample + dist.probs - dist.probs.detach() 
    
    flat_stoch = torch.flatten(stoch, start_dim=-2, end_dim=-1)
    
    return dist, flat_stoch

# ==========================================
# 2. RSSM
# ==========================================
class RSSM(nn.Module):
    def __init__(self, action_dim=3, stoch_dim=64, discrete_dim=32, determ_dim=512, hidden_dim=512):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        self.determ_dim = determ_dim

        # 과거의 상태(z)와 행동(a)을 받아 현재의 관성(h)을 만듦
        self.gru_input = nn.Sequential(
            nn.Linear(stoch_dim * discrete_dim + action_dim, hidden_dim),
            nn.ELU()
        )
        self.gru = nn.GRUCell(hidden_dim, determ_dim)

        # Prior (상상): 관성(h)만으로 미래의 상태(z)를 예측
        self.prior_net = nn.Sequential(
            nn.Linear(determ_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * discrete_dim)
        )
        
        # Posterior (현실): 관성(h)과 실제 관측(obs)을 합쳐 현재의 상태(z)를 확정
        self.post_net = nn.Sequential(
            nn.Linear(determ_dim + 1024, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * discrete_dim)
        )

    def forward(self, prev_z, prev_action, prev_h, obs_embed=None):
        # 1. Deterministic state (h) 업데이트
        x = self.gru_input(torch.cat([prev_z, prev_action], dim=-1))
        h = self.gru(x, prev_h)

        # 2. Prior 예측
        prior_logits = self.prior_net(h)
        prior_dist, prior_z = get_categorical_state(prior_logits, self.stoch_dim, self.discrete_dim)

        # 3. Posterior 확정
        if obs_embed is not None:
            post_logits = self.post_net(torch.cat([h, obs_embed], dim=-1))
            post_dist, post_z = get_categorical_state(post_logits, self.stoch_dim, self.discrete_dim)
            return h, post_z, prior_logits, post_logits
        else:
            return h, prior_z, prior_logits, None

# ==========================================
# 3. 인코더 & 디코더
# ==========================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024)
        )

    def forward(self, obs):
        return self.net(obs)

class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2, padding=2, output_padding=1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 6, 2, padding=2), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 6, 2, padding=2) 
        )

    def forward(self, latent):
        x = self.fc(latent).view(-1, 1024, 4, 4)
        pred = self.net(x)
        dist = Normal(pred, 0.1)
        return Independent(dist, 3)

class RewardModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent):
        pred = self.net(latent)
        dist = Normal(pred, 1.0)
        return Independent(dist, 1)

class ContinueModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent):
        logits = self.net(latent)
        return Bernoulli(logits=logits)

# ==========================================
# 4. 통합 World Model
# ==========================================
class WorldModel(nn.Module):
    def __init__(self, action_dim=3, stoch_dim=64, discrete_dim=32, determ_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.rssm = RSSM(action_dim, stoch_dim, discrete_dim, determ_dim)
        
        self.latent_dim = determ_dim + (stoch_dim * discrete_dim) # 512 + 2048 = 2560
        
        self.observation_decoder = ObservationDecoder(self.latent_dim)
        self.predict_reward = RewardModel(self.latent_dim)
        self.predict_continue = ContinueModel(self.latent_dim)

    def forward(self, obs, action, prev_state=None):
        device = obs.device
        B, T, C, H, W = obs.shape

        if prev_state is None:
            prev_z = torch.zeros(B, self.rssm.stoch_dim * self.rssm.discrete_dim, device=device)
            prev_h = torch.zeros(B, self.rssm.determ_dim, device=device)
        else:
            prev_h, prev_z = prev_state

        hs, zs = [], []
        prior_logits_list, post_logits_list = [], []

        obs_embeds = self.encoder(obs.view(B*T, C, H, W)).view(B, T, -1)

        shifted_action = torch.cat([torch.zeros(B, 1, action.shape[-1], device=device), action[:, :-1]], dim=1)

        for t in range(T):
            h, z, prior_logits, post_logits = self.rssm(
                prev_z, shifted_action[:, t], prev_h, obs_embeds[:, t]
            )
            hs.append(h)
            zs.append(z)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)

            prev_h, prev_z = h, z

        hs = torch.stack(hs, dim=1)
        zs = torch.stack(zs, dim=1)
        prior_logits = torch.stack(prior_logits_list, dim=1)
        post_logits = torch.stack(post_logits_list, dim=1)

        return hs, zs, prior_logits, post_logits