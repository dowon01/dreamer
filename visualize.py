import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.world_model import WorldModel
from utils.buffer import ReplayBuffer

def visualize_reconstruction(model_path, data_dir="data"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. 모델 로드
    world_model = WorldModel().to(device)
    world_model.load_state_dict(torch.load(model_path, map_location={'cpu' : 'mps'}))
    world_model.eval()

    # 2. 데이터 샘플링 (검증 데이터셋에서)
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    valid_files = all_files[-2:] # 마지막 2개 에피소드 사용
    buffer = ReplayBuffer(data_dir=data_dir, files=valid_files, seq_len=10, batch_size=1)
    obs, action, reward, done = buffer.sample_batch()
    obs, action = obs.to(device), action.to(device)

    # 3. 모델의 복원 결과 확인
    with torch.no_grad():
        hs, zs, _, _ = world_model(obs, action)
        latent = torch.cat([hs, zs], dim=-1)
        # 시간축(T)을 펼쳐서 복원
        B, T, _ = latent.shape
        rec_obs = world_model.observation_decoder(latent.view(B*T, -1)).mean.view(B, T, 3, 64, 64)

    # 4. 시각화 (원본 vs 복원)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # 원본 (위)
        orig = obs[0, i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        # 복원 (아래)
        rec = rec_obs[0, i].permute(1, 2, 0).cpu().numpy()
        rec = np.clip(rec, 0, 1) # 이미지 값 범위 고정
        axes[1, i].imshow(rec)
        axes[1, i].set_title(f"Dream {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    print("시각화 완료!")

if __name__ == "__main__":
    visualize_reconstruction("output/wm_iter_225.pth")