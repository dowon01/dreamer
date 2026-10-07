import os
import random
import numpy as np
import torch

# 에피소드 저장 형식 (길이 N+1)
#   obs[t]    : t 시점 관측
#   action[t] : obs[t]에 도달하게 만든 행동 (action[0] = 0)
#   reward[t] : obs[t]에 도달하며 받은 보상 (reward[0] = 0)
#   done[t]   : obs[t]가 종료 상태인지 (시간 제한은 제외)
def save_episode(path, obs, action, reward, done):
    np.savez(path,
             obs=np.array(obs, dtype=np.uint8),
             action=np.array(action, dtype=np.float32),
             reward=np.array(reward, dtype=np.float32),
             done=np.array(done, dtype=np.float32))

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

        current_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        new_files = [f for f in current_files if f not in self.loaded_files]
        # 오래된 순으로 정렬 후, 어차피 버려질 파일은 읽지 않음
        new_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.data_dir, f)))
        skipped = new_files[:-self.max_episodes]
        new_files = new_files[-self.max_episodes:]
        self.loaded_files.update(skipped)

        for fname in new_files:
            file_path = os.path.join(self.data_dir, fname)
            try:
                with np.load(file_path) as data:
                    episode = {k: data[k].copy() for k in ('obs', 'action', 'reward', 'done')}

                self.loaded_files.add(fname)
                if len(episode['obs']) > self.seq_len:
                    self.episodes.append(episode)
            except Exception as e:
                print(f"파일 로드 실패 ({fname}): {e}")

        while len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def _sample_sequence(self):
        if not self.episodes:
            raise ValueError(f"버퍼에 seq_len({self.seq_len})보다 긴 에피소드가 없습니다")

        ep = random.choice(self.episodes)
        length = len(ep['obs'])

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
        # 새 에피소드 반영은 수집 직후 load_new_data()를 직접 호출 (매 스텝 디렉토리 스캔 방지)
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for _ in range(self.batch_size):
            o, a, r, d = self._sample_sequence()
            obs_batch.append(o)
            act_batch.append(a)
            rew_batch.append(r)
            done_batch.append(d)

        obs_tensor = torch.FloatTensor(np.array(obs_batch)).permute(0, 1, 4, 2, 3)
        obs_tensor = obs_tensor / 255.0 - 0.5

        act_tensor = torch.FloatTensor(np.array(act_batch))
        rew_tensor = torch.FloatTensor(np.array(rew_batch)).unsqueeze(-1)
        done_tensor = torch.FloatTensor(np.array(done_batch)).unsqueeze(-1)

        return obs_tensor, act_tensor, rew_tensor, done_tensor
