import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from models.world_model import WorldModel
from models.actor_critic import Actor

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=4):
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

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 환경 생성 및 ActionRepeat 적용
    env = gym.make("CarRacing-v3", render_mode="human")
    env = ActionRepeat(env, repeat=4)
    
    world_model = WorldModel().to(device)
    actor = Actor(latent_dim=2560).to(device) 
    
    # 최신 체크포인트 로드
    world_model.load_state_dict(torch.load("output/wm_iter_2150.pth", map_location=device))
    actor.load_state_dict(torch.load("output/actor_iter_2150.pth", map_location=device))
    
    world_model.eval()
    actor.eval()

    obs, _ = env.reset()
    prev_h = torch.zeros(1, 512).to(device)
    prev_z = torch.zeros(1, 2048).to(device)
    prev_action = torch.zeros(1, 3).to(device)

    print("실전 주행 시작")
    
    total_eval_reward = 0.0

    with torch.no_grad():
        while True:
            # 관측값 정규화 ([-0.5, 0.5] 범위 맞추기)
            obs_tensor = torch.FloatTensor(obs.copy()).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            obs_tensor = F.interpolate(obs_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            obs_tensor = obs_tensor - 0.5
            
            # 현재 상태 업데이트
            embed = world_model.encoder(obs_tensor)
            h, z, _, _ = world_model.rssm(prev_z, prev_action, prev_h, embed)
            
            # 행동 결정
            latent = torch.cat([h, z], dim=-1)
            action_tensor = actor(latent, deterministic=True)
            action_np = action_tensor.cpu().numpy()[0]
            
            # 규격 변환 ([-1, 1] -> [0, 1])
            env_action = np.array([
                action_np[0],                # Steering: [-1, 1]
                (action_np[1] + 1.0) / 2.0,  # Gas: [0, 1]
                (action_np[2] + 1.0) / 2.0   # Brake: [0, 1]
            ])
            env_action = np.clip(env_action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            
            # 환경 적용
            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_eval_reward += reward
            
            # 다음 스텝을 위해 prev_action은 신경망이 뱉은 원본 텐서를 줘야 함
            prev_h, prev_z, prev_action = h, z, action_tensor

            print(f"Action -> Steering: {env_action[0]:.2f}, Gas: {env_action[1]:.2f}, Brake: {env_action[2]:.2f} | Reward: {reward:.2f}")
            
            if terminated or truncated:
                print(f"최종 점수: {total_eval_reward:.2f}")
                break

    env.close()

if __name__ == "__main__":
    evaluate()