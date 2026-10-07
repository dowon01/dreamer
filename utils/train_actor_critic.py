import torch

HORIZON = 15             # 상상 미래 길이
GAMMA = 0.99             # 미래 보상 할인율
LAMBDA = 0.95            # Lambda-return 계수
ENTROPY_COEFF = 3e-4     # 탐험을 위한 엔트로피 가중치

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

class ReturnNormalizer:
    # 수익의 5~95 백분위 범위를 EMA로 추적해 actor loss의 스케일을 일정하게 유지
    def __init__(self, decay=0.99, low=0.05, high=0.95):
        self.decay, self.low, self.high = decay, low, high
        self.scale = None

    def update(self, returns):
        flat = returns.detach().flatten().float()
        q = torch.quantile(flat, torch.tensor([self.low, self.high], device=flat.device))
        value = (q[1] - q[0]).item()
        self.scale = value if self.scale is None else self.decay * self.scale + (1 - self.decay) * value
        return max(1.0, self.scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = state["scale"]

# ==========================================
# 2. Actor-Critic 학습 (상상 주행)
# ==========================================
def train_actor_critic(world_model, actor, critic, target_critic, actor_opt, critic_opt, start_hs, start_zs, device, return_norm):
    world_model.eval()

    start_h = start_hs.reshape(-1, start_hs.shape[-1]) # (B*T, 512)
    start_z = start_zs.reshape(-1, start_zs.shape[-1]) # (B*T, 2048)

    # 상상 주행 (월드 모델의 Prior만으로 다음 상태를 예측)
    with torch.no_grad():
        imag_h, imag_z, imag_samples = [start_h], [start_z], []
        curr_h, curr_z = start_h, start_z
        for _ in range(HORIZON):
            sample = actor.get_dist(torch.cat([curr_h, curr_z], dim=-1)).sample()
            curr_h, curr_z, _, _ = world_model.rssm(curr_z, sample.clamp(-1.0, 1.0), curr_h, None)
            imag_h.append(curr_h)
            imag_z.append(curr_z)
            imag_samples.append(sample)

        imag_latents = torch.cat([torch.stack(imag_h), torch.stack(imag_z)], dim=-1) # (HORIZON+1, B*T, 2560)
        imag_samples = torch.stack(imag_samples)

        # 행동의 결과(보상, 종료 여부)는 그 행동으로 도달한 다음 상태에서 예측
        imag_rewards = world_model.predict_reward(imag_latents[1:]).mean
        imag_continues = world_model.predict_continue(imag_latents[1:]).mean
        imag_values_target = target_critic(imag_latents[1:]).mean

        # Lambda Return 계산
        targets = compute_lambda_return(imag_rewards, imag_values_target, imag_continues)

    # Critic
    curr_values_dist = critic(imag_latents[:-1])
    critic_loss = -curr_values_dist.log_prob(targets).mean()

    # Actor (REINFORCE): advantage = (lambda-return - V) / 수익 범위
    ret_scale = return_norm.update(targets)
    advantage = ((targets - curr_values_dist.mean.detach()) / ret_scale).squeeze(-1)
    dist = actor.get_dist(imag_latents[:-1])
    entropy = dist.entropy()
    actor_loss = -(dist.log_prob(imag_samples) * advantage).mean() - ENTROPY_COEFF * entropy.mean()

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
        "entropy": entropy.mean().item(),
        "target_mean": targets.mean().item(),
        "ret_scale": ret_scale
    }
