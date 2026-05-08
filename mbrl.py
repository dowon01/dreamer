import gymnasium as gym
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from utils.buffer import ReplayBuffer
from utils.train_world_model import train_world_model 
from utils.train_actor_critic import train_actor_critic

# 환경의 시간을 압축해주는 Action Repeat Wrapper
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

def collect_episode(env, world_model, actor, device, iteration, data_dir="data/mbrl", is_eval=False):
    obs, _ = env.reset()
    done = False
    
    episode_obs, episode_act, episode_rew, episode_done = [], [], [], []
    
    prev_h = torch.zeros(1, 512).to(device)
    prev_z = torch.zeros(1, 2048).to(device)
    prev_action = torch.zeros(1, 3).to(device)
    
    total_reward = 0
    step_count = 0

    prev_steer = 0.0 # 조향각 변수 저장

    while not done:
        # 이미지 전처리: 96x96 -> 64x64 및 [-0.5, 0.5] 정규화
        obs_tensor = torch.FloatTensor(obs.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
        obs_64 = F.interpolate(obs_tensor / 255.0, size=(64, 64)) - 0.5
        
        with torch.no_grad():
            embed = world_model.encoder(obs_64)
            h, z, _, _ = world_model.rssm(prev_z, prev_action, prev_h, embed)
            
            action_tensor = actor(torch.cat([h, z], dim=-1), deterministic=is_eval)
            action = action_tensor.cpu().numpy()[0]

            # 신경망의 출력([-1, 1])을 환경의 규격에 맞게 변환 -> result 결과에 사용함
            # env_action = np.array([
            #     action[0],                # Steering: [-1, 1] 그대로
            #     (action[1] + 1.0) / 2.0,  # Gas: [-1, 1] -> [0, 1] 로 변환
            #     (action[2] + 1.0) / 2.0   # Brake: [-1, 1] -> [0, 1] 로 변환
            # ])
            env_action = np.array([
                action[0],  # Steering: [-1, 1] 
                action[1],  # Gas: [-1, 1]
                action[2]   # Brake: [-1, 1]
            ])

            env_action = np.clip(env_action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        # 환경 한 스텝 진행
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        # 조향각 미세 조정 -> 핸들을 계속 꺾는 현상을 줄여, 자연스러운 주행을 위해
        if not is_eval: # 훈련 중에만 페널티를 주어 습관을 고침
            current_steer = action[0] # 현재 조향 (-1 ~ 1)
            current_gas = action[1]   # 현재 엑셀 (-1 ~ 1, 실제론 0 이상일때 가속)
            
            # 핸들을 확 꺾으면 감점 (이전 각도와의 차이)
            steer_penalty = 0.1 * abs(current_steer - prev_steer)
            
            # 브레이크 유도: 핸들을 크게 꺾었는데 엑셀도 밟고 있으면 감점
            corner_penalty = 0.05 * abs(current_steer) * max(0, current_gas)
            
            # 원래 보상에서 페널티들을 깎아버림
            reward = reward - steer_penalty - corner_penalty
            
            prev_steer = current_steer # 다음 프레임을 위해 저장
        
        # 데이터 수집
        if not is_eval:
            # 원본 이미지(obs_64)는 이미 [-0.5, 0.5]이므로 다시 [0, 255]의 uint8로 복구하여 저장
            resized_obs = ((obs_64.squeeze(0).permute(1, 2, 0).cpu().numpy() + 0.5) * 255).astype(np.uint8)
            episode_obs.append(resized_obs)
            episode_act.append(action)
            episode_rew.append(reward / 5.0) # 보상 스케일링
            episode_done.append(done)
        
        obs = next_obs
        prev_h, prev_z, prev_action = h, z, torch.FloatTensor(action).unsqueeze(0).to(device)
        total_reward += reward
        step_count += 1

    if not is_eval:
        timestamp = int(datetime.datetime.now().timestamp())
        np.savez(f"{data_dir}/episode{timestamp}.npz", obs=episode_obs, action=episode_act, reward=episode_rew, done=episode_done)
    
    return total_reward, step_count

def collect_episode_random(env, data_dir="data/mbrl"):
    obs, _ = env.reset()
    done = False
    episode_obs, episode_act, episode_rew, episode_done = [], [], [], []
    
    while not done:
        action = env.action_space.sample() 
        obs_tensor = torch.FloatTensor(obs.copy()).permute(2, 0, 1).unsqueeze(0)
        obs_64 = F.interpolate(obs_tensor / 255.0, size=(64, 64)) - 0.5
        resized_obs = ((obs_64.squeeze(0).permute(1, 2, 0).numpy() + 0.5) * 255).astype(np.uint8)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_obs.append(resized_obs)
        episode_act.append(action)
        episode_rew.append(reward / 5.0)
        episode_done.append(done)
        obs = next_obs

    os.makedirs(data_dir, exist_ok=True)
    timestamp = int(datetime.datetime.now().timestamp())
    np.savez(f"{data_dir}/seed_{timestamp}.npz", 
             obs=np.array(episode_obs), action=np.array(episode_act), 
             reward=np.array(episode_rew), done=np.array(episode_done))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # 에이전트의 1스텝 = 실제 물리엔진의 4프레임
    env = ActionRepeat(env, repeat=4)

    log_dir = f"runs/dreamer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 시작: tensorboard --logdir runs")
    
    world_model = WorldModel().to(device)
    actor = Actor(latent_dim=2560).to(device)
    critic = Critic(latent_dim=2560).to(device)

    def initialize_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    world_model.apply(initialize_weights)
    actor.apply(initialize_weights)
    critic.apply(initialize_weights)

    world_model.load_state_dict(torch.load("output/wm_iter_8000.pth", map_location=device))
    actor.load_state_dict(torch.load("output/actor_iter_8000.pth", map_location=device))
    critic.load_state_dict(torch.load("output/critic_iter_8000.pth", map_location=device))
    
    # 1 - 7000, 2 - 3000 Iteration에서 학습률을 낮추어 재개
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=2e-4) # 2e-4 -> 1e-4로 감소
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4) # 1e-4 -> 2e-5로 감소
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4) # 1e-4 -> 2e-5로 감소

    target_critic = Critic(latent_dim=2560).to(device)
    target_critic.load_state_dict(critic.state_dict())

    os.makedirs("data/mbrl", exist_ok=True)
    if len(os.listdir("data/mbrl")) < 50:
        print("시드 데이터 수집 중...")
        for _ in tqdm(range(50), desc="Random Seed"):
            collect_episode_random(env, data_dir="data/mbrl")

    # 배치 사이즈를 32
    buffer = ReplayBuffer("data/mbrl", seq_len=50, batch_size=32)

    print("\nMBRL start")
    global_step = 8001

    for iteration in range(8001, 10001):
        print(f"\n=== Iteration {iteration} ===")
        
        # 1. 수집
        train_reward, step_count = collect_episode(env, world_model, actor, device, iteration, is_eval=False)
        writer.add_scalar("Rollout/Train_Reward", train_reward, iteration)
        writer.add_scalar("Rollout/Train_Step_Count", step_count, iteration)
        print(f"[훈련 보상] {train_reward:.2f}")

        # 2. Update-To-Data(생존 스텟 비례 학습)
        num_updates = int(step_count / 2) # 살아남은 프레임 수의 절반에 비례하여 업데이트 횟수 결정
        pbar = tqdm(range(num_updates), desc="   Training (WM & AC)", leave=False)
        for step in pbar:
            batch = buffer.sample_batch()
            
            # World Model 학습
            wm_loss, detached_hs, detached_zs = train_world_model(world_model, wm_opt, batch, device, is_train=True)
            
            # Actor-Critic 학습
            ac_loss = train_actor_critic(world_model, actor, critic, target_critic, actor_opt, critic_opt, detached_hs, detached_zs, device)
            
            global_step += 1
            
            # 20 스텝마다 텐서보드 및 터미널 기록
            if step % 20 == 0:
                pbar.set_postfix({
                    "WM_Obs": f"{wm_loss['obs']:.4f}",
                    "WM_KL": f"{wm_loss['kl']:.2f}",
                    "Actor": f"{ac_loss['actor_loss']:.3f}"
                })
                
                for k, v in wm_loss.items(): writer.add_scalar(f"Loss/WM_{k}", v, global_step)
                for k, v in ac_loss.items(): writer.add_scalar(f"Loss/AC_{k}", v, global_step)
                writer.add_scalar("Target_Mean", ac_loss["target_mean"], global_step)

        # 3. 평가 (10 Iteration 마다)
        if iteration % 10 == 0:
            eval_reward, step_count = collect_episode(env, world_model, actor, device, iteration, is_eval=True)
            writer.add_scalar("Rollout/Eval_Reward", eval_reward, iteration)
            writer.add_scalar("Rollout/Eval_Step_Count", step_count, iteration)
            print(f"[실전 평가 보상] {eval_reward:.2f} (노이즈 제거)")

        # 4. 모델 저장 및 시각화 (50 Iteration 마다)
        if iteration % 50 == 0:
            os.makedirs("output", exist_ok=True)
            torch.save(actor.state_dict(), f"output/actor_iter_{iteration}.pth")
            torch.save(world_model.state_dict(), f"output/wm_iter_{iteration}.pth")
            torch.save(critic.state_dict(), f"output/critic_iter_{iteration}.pth")
            
            batch = buffer.sample_batch()
            obs = batch[0].to(device) # (B, T, C, H, W)
            action = batch[1].to(device)
            
            with torch.no_grad():
                hs, zs, _, _ = world_model(obs, action)
                latent = torch.cat([hs[:, 0], zs[:, 0]], dim=-1) # 첫 번째 프레임만 시각화
                recon_dist = world_model.observation_decoder(latent)
                recon = recon_dist.mean # [-0.5, 0.5] 범위
                
                # 시각화를 위해 [0, 1] 범위로 복구
                orig_img = obs[:, 0] + 0.5 
                recon_img = recon + 0.5
                
                grid = torch.cat([orig_img, recon_img], dim=-1) 
                writer.add_images("Visual/Real_vs_Recon", grid[:4], iteration)
            
            print(f"모델 저장 & 시각화 완료")