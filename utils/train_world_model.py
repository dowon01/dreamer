import torch
import torch.nn.functional as F

FREE_BITS = 0.1         # 최소 정보량 하한선
OBS_WEIGHT = 10.0       # 시야를 선명하게 만들기 위한 강력한 압박
KL_ALPHA = 0.8           # KL Balancing 비율 (Prior 80%, Posterior 20%)

# ==========================================
# 유틸리티 함수
# ==========================================
def kl_balancing_categorical(post_logits, prior_logits):
    """Prior가 80%의 힘으로 쫓아가고, Posterior가 20%만 양보하는 완벽한 균형"""
    post_probs_det = F.softmax(post_logits.detach(), dim=-1)
    prior_probs = F.softmax(prior_logits, dim=-1)
    kl_prior = post_probs_det * (torch.log(post_probs_det + 1e-8) - torch.log(prior_probs + 1e-8))
    loss_prior = kl_prior.sum(dim=-1).mean()

    post_probs = F.softmax(post_logits, dim=-1)
    prior_probs_det = F.softmax(prior_logits.detach(), dim=-1)
    kl_post = post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs_det + 1e-8))
    loss_post = kl_post.sum(dim=-1).mean()

    return KL_ALPHA * loss_prior + (1 - KL_ALPHA) * loss_post

# ==========================================
# World Model 학습 (눈 뜨기)
# ==========================================
def train_world_model(world_model, optimizer, batch, device, is_train=True):
    obs, action, reward, done = [x.to(device) for x in batch]
    B, T, C, H, W = obs.shape
    
    if is_train:
        world_model.train()
        hs, zs, prior_logits, post_logits = world_model(obs, action)
    else:
        world_model.eval()
        with torch.no_grad():
            hs, zs, prior_logits, post_logits = world_model(obs, action)
    
    latent = torch.cat([hs, zs], dim=-1) # (B, T, 2560)

    # 1. 복원 Loss (Observation)
    dist_obs = world_model.observation_decoder(latent.view(B*T, -1))
    log_prob_obs = dist_obs.log_prob(obs.view(B*T, C, H, W)).view(B, T)
    
    # 커브 가중치 (핸들을 꺾었을 때 지형 복원을 더 꼼꼼히 하도록 강제)
    steering = torch.abs(action[:, :, 0]) # (B, T)
    curve_weight = 1.0 + steering * 4.0 
    
    loss_obs = -(log_prob_obs * curve_weight).mean() / (C * H * W)

    # 2. 보상 및 종료 예측 Loss
    # 이전 시간(t-1)의 잠재 상태로 현재(t)를 예측하므로 1: 부터 슬라이싱
    reward_dist = world_model.predict_reward(latent[:, 1:])
    loss_reward = -reward_dist.log_prob(reward[:, 1:]).mean()

    continue_dist = world_model.predict_continue(latent[:, 1:])
    loss_continue = -continue_dist.log_prob(1.0 - done[:, 1:]).mean() * 5.0
    
    # 3. KL Loss
    post_dist = post_logits.view(B, T, 64, 32)[:, 1:]
    prior_dist = prior_logits.view(B, T, 64, 32)[:, 1:]
    
    loss_kl_raw = kl_balancing_categorical(post_dist, prior_dist)
    loss_kl = torch.max(loss_kl_raw, torch.tensor(FREE_BITS).to(device))
    
    # 4. 전체 Loss 합산
    total_loss = OBS_WEIGHT * loss_obs + loss_reward + loss_continue + 1.0 * loss_kl
    
    if is_train:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
        optimizer.step()
    
    return {
        "total": total_loss.item(),
        "obs": loss_obs.item(),
        "reward": loss_reward.item(),
        "continue": loss_continue.item(),
        "kl": loss_kl.item()
    }, hs.detach(), zs.detach()