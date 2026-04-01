import torch
import torch.nn.functional as F

HORIZON = 8           # 상상 미래 길이 (원본 논문 권장치 15)
GAMMA = 0.997            # 미래 보상 할인율
LAMBDA = 0.95            # Lambda-return 계수
ENTROPY_COEFF = 1e-4     # 탐험을 위한 엔트로피 가중치
REINFORCE_COEF = 0.0     # REINFORCE(간접 학습) 비율

# ==========================================
# 유틸리티 함수
# ==========================================
def compute_lambda_return(rewards, values, continues):
    # 미래 가치를 현재로 끌어오는 TD(lambda) 계산
    T = rewards.shape[0]
    returns = torch.zeros_like(values)

    last_return = values[-1]

    for t in reversed(range(T)):
        last_return = rewards[t] + GAMMA * continues[t] * (
            (1 - LAMBDA) * values[t] + LAMBDA * last_return
        )
        returns[t] = last_return

    return returns

# ==========================================
# 2. Actor-Critic 학습 (상상 주행)
# ==========================================
def train_actor_critic(world_model, actor, critic, target_critic, actor_opt, critic_opt, start_hs, start_zs, device):
    world_model.eval()
        
    start_h = start_hs.reshape(-1, start_hs.shape[-1]) # (B*T, 512)
    start_z = start_zs.reshape(-1, start_zs.shape[-1]) # (B*T, 2048)

    imag_h, imag_z = [start_h], [start_z]
    imag_actions, imag_log_probs, imag_entropy = [], [], []
    curr_h, curr_z = start_h, start_z

    # 상상 주행
    for _ in range(HORIZON):
        latent = torch.cat([curr_h, curr_z], dim=-1)
        action, log_prob = actor.get_action_and_log_prob(latent)
        
        # 월드 모델의 Prior 신경망을 통해 다음 상태를 상상
        curr_h, curr_z, _, _ = world_model.rssm(curr_z, action, curr_h, None)

        imag_h.append(curr_h)
        imag_z.append(curr_z)
        imag_actions.append(action)
        imag_log_probs.append(log_prob)

        # 엔트로피 계산
        imag_entropy.append(-log_prob)

    imag_hs = torch.stack(imag_h, dim=0)
    imag_zs = torch.stack(imag_z, dim=0)
    imag_log_probs = torch.stack(imag_log_probs, dim=0)
    imag_entropy = torch.stack(imag_entropy, dim=0)
    imag_latents = torch.cat([imag_hs, imag_zs], dim=-1) # (HORIZON+1, B*T, 2560)

    # 보상 및 가치 예측 
    imag_reward_dist = world_model.predict_reward(imag_latents[:-1])
    imag_rewards = imag_reward_dist.mean
    
    imag_continues = torch.ones_like(imag_rewards)

    with torch.no_grad():
        imag_values_target_dist = target_critic(imag_latents[1:])
        imag_values_target = imag_values_target_dist.mean
        
    # Lambda Return 계산
    targets = compute_lambda_return(imag_rewards, imag_values_target, imag_continues)
    
    curr_values_dist = critic(imag_latents[:-1].detach())
    
    critic_loss = -curr_values_dist.log_prob(targets.detach()).mean()
    
    dynamics_loss = -targets.mean()

    actor_loss = dynamics_loss - ENTROPY_COEFF * imag_entropy.mean()

    # 통합 업데이트
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    
    actor_loss.backward()
    critic_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
    
    actor_opt.step()
    critic_opt.step()

    # Target Network Soft Update (0.98)
    with torch.no_grad():
        for p, p_target in zip(critic.parameters(), target_critic.parameters()):
            p_target.data.copy_(0.98 * p_target.data + 0.02 * p.data)

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": imag_entropy.mean().item(),
        "target_mean": targets.mean().item()
    }