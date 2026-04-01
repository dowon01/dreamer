import gymnasium as gym
import numpy as np
import cv2

class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 이미지는 64x64로 리사이즈하여 모델의 부하를 줄임
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def step(self, action):
        # 1. 환경에서 한 스텝 진행
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. 전처리: 이미지를 64x64로 축소하고 정규화 준비
        obs = self._preprocess_obs(obs)
        
        # 3. 종료 신호 통합 (Terminated 혹은 Truncated 시 True)
        done = terminated or truncated
        
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def _preprocess_obs(self, obs):
        # 상단 점수판 제거 및 64x64 리사이즈
        # CarRacing-v3 이미지 규격에 맞춰 슬라이싱 (점수판 제외)
        obs = obs[:84, :, :] 
        obs = cv2.resize(obs, (64, 64))
        return obs

def make_env(render_mode="rgb-array"):
    # CarRacing-v3 환경 생성
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = CarRacingWrapper(env)
    return env