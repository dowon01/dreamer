import gymnasium as gym
import numpy as np
import torch

OBS_SIZE = 64
ACTION_REPEAT = 4
REWARD_SCALE = 5.0      # 버퍼에 저장할 때 보상을 이 값으로 나눔

# 환경의 시간을 압축해주는 Action Repeat Wrapper
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=ACTION_REPEAT):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

def make_env(render_mode="rgb_array", repeat=ACTION_REPEAT):
    # 에이전트의 1스텝 = 실제 물리엔진의 4프레임
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    return ActionRepeat(env, repeat=repeat)

# ==========================================
# 관측 전처리
# ==========================================
def resize_obs(obs):
    # 96x96x3 -> 64x64x3 (nearest)
    h, w = obs.shape[:2]
    rows = (np.arange(OBS_SIZE) * h / OBS_SIZE).astype(np.int64)
    cols = (np.arange(OBS_SIZE) * w / OBS_SIZE).astype(np.int64)
    return obs[rows][:, cols]

def obs_to_tensor(obs, device):
    # 64x64x3 uint8 -> (1, 3, 64, 64) float, [-0.5, 0.5] 정규화
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return (obs_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0) - 0.5

# ==========================================
# 액션 변환 (Actor 출력 [-1, 1]^3 -> 환경 규격)
# ==========================================
def to_env_action(action):
    # Steering: [-1, 1] 그대로, Gas/Brake: 음수는 0으로 자름
    return np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

def sample_random_action():
    # 시드 데이터 수집용 (Actor와 같은 [-1, 1] 범위)
    return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
