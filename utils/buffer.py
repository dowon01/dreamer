import os
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, data_dir, seq_len=50, batch_size=16, max_episodes=500):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        
        self.episodes = [] 
        self.loaded_files = set()
        
        self.load_new_data()

    def load_new_data(self):
        if not os.path.exists(self.data_dir):
            return
            
        current_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        new_files = [f for f in current_files if f not in self.loaded_files]
        
        for fname in new_files:
            file_path = os.path.join(self.data_dir, fname)
            try:
                data = np.load(file_path)
                episode = {
                    'obs': data['obs'].copy(),
                    'action': data['action'].copy(),
                    'reward': data['reward'].copy(),
                    'done': data['done'].copy()
                }
                data.close()
                
                self.episodes.append(episode)
                self.loaded_files.add(fname)
            except Exception as e:
                print(f"파일 로드 실패 ({fname}): {e}")
        
        while len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def _sample_sequence(self):
        if not self.episodes:
            raise ValueError("버퍼에 데이터가 없습니다!")
            
        ep = random.choice(self.episodes)
        length = len(ep['obs'])
        
        if length <= self.seq_len:
            return self._sample_sequence()

        start_idx = 0
        for _ in range(20):
            start_idx = np.random.randint(0, length - self.seq_len)
            action_seq = ep['action'][start_idx : start_idx + self.seq_len]
            
            steering_intensity = np.mean(np.abs(action_seq[:, 0]))
            if steering_intensity > 0.1:
                break

        obs = ep['obs'][start_idx : start_idx + self.seq_len]
        action = ep['action'][start_idx : start_idx + self.seq_len]
        reward = ep['reward'][start_idx : start_idx + self.seq_len]
        done = ep['done'][start_idx : start_idx + self.seq_len]
        
        return obs, action, reward, done

    def sample_batch(self):
        self.load_new_data()
        
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for _ in range(self.batch_size):
            o, a, r, d = self._sample_sequence()
            obs_batch.append(o)
            act_batch.append(a)
            rew_batch.append(r)
            done_batch.append(d)
            
        obs_tensor = torch.FloatTensor(np.array(obs_batch)).permute(0, 1, 4, 2, 3)
        
        # 디코더에서 Sigmoid를 뺐으므로 [-0.5, 0.5]로 맞춤
        obs_tensor = obs_tensor / 255.0 - 0.5
        
        act_tensor = torch.FloatTensor(np.array(act_batch))
        rew_tensor = torch.FloatTensor(np.array(rew_batch)).unsqueeze(-1)
        done_tensor = torch.FloatTensor(np.array(done_batch)).unsqueeze(-1)
        
        return obs_tensor, act_tensor, rew_tensor, done_tensor