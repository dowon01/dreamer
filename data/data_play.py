import pygame
from dreamer.data.dataset import make_env
import numpy as np
import os

def get_action_from_keyboard():
    action = np.array([0.0, 0.0, 0.0])
    keys = pygame.key.get_pressed()
    # 왼쪽/오른쪽 (스티어링)
    if keys[pygame.K_LEFT]:  action[0] = -1.0
    elif keys[pygame.K_RIGHT]: action[0] = 1.0
    # 위 (가속)
    if keys[pygame.K_UP]:    action[1] = 1.0
    # 아래 (브레이크)
    if keys[pygame.K_DOWN]:  action[2] = 0.8
    return action

def collect_expert_data(total_episodes=20):
    env = make_env(render_mode="human")
    pygame.init()
    pygame.display.set_mode((100, 100)) 
    
    save_dir = "data"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for eps in range(total_episodes):
        obs, info = env.reset()
        done = False
        episode_data = {'obs': [], 'action': [], 'reward': [], 'done': []}
        
        print(f"Episode {eps+1} 시작! 'CarRacing' 창을 클릭하고 방향키로 운전하세요.")

        while not done:
            pygame.event.pump()
            
            action = get_action_from_keyboard()
            next_obs, reward, done, info = env.step(action)
            
            
            episode_data['obs'].append(obs)
            episode_data['action'].append(action)
            episode_data['reward'].append(reward)
            episode_data['done'].append(float(done))
            
            obs = next_obs

        np.savez(f"{save_dir}/eps_{eps+1}.npz", **episode_data)
        print(f"Episode {eps+1} 완료!")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    collect_expert_data()