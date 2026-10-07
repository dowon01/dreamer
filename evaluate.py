import argparse
import os
import numpy as np
import torch
import imageio
from models.world_model import WorldModel
from models.actor_critic import Actor
from utils.checkpoint import checkpoint_paths, resolve_tag
from utils.env import make_env, resize_obs, obs_to_tensor, to_env_action

def evaluate(ckpt="latest", output_dir="output", render=True, video_path="videos/eval_run.mp4", fps=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(render_mode="rgb_array")

    world_model = WorldModel().to(device)
    actor = Actor(latent_dim=world_model.latent_dim).to(device)

    paths = checkpoint_paths(output_dir, resolve_tag(output_dir, ckpt))
    world_model.load_state_dict(torch.load(paths['wm'], map_location=device))
    actor.load_state_dict(torch.load(paths['actor'], map_location=device))
    print(f"체크포인트 로드: {paths['wm']}, {paths['actor']}")

    world_model.eval()
    actor.eval()

    obs, _ = env.reset()
    prev_h, prev_z = world_model.initial_state(1, device)
    prev_action = torch.zeros(1, 3).to(device)

    print("실전 주행 시작")

    total_eval_reward = 0.0
    frames = [] # 비디오 프레임을 담을 빈 상자

    if render:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Dreamer Eval (WSL Safe)")

    with torch.no_grad():
        while True:
            if render and any(event.type == pygame.QUIT for event in pygame.event.get()):
                print("창이 닫혀 주행을 중단합니다")
                break

            # 관측값 전처리 (학습과 동일한 64x64 리사이즈 및 [-0.5, 0.5] 정규화)
            embed = world_model.encoder(obs_to_tensor(resize_obs(obs), device))
            h, z, _, _ = world_model.rssm(prev_z, prev_action, prev_h, embed)

            # 행동 결정
            latent = torch.cat([h, z], dim=-1)
            action_tensor = actor(latent, deterministic=True)
            env_action = to_env_action(action_tensor.cpu().numpy()[0])

            # 환경 적용
            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_eval_reward += reward

            # 프레임 수집 (+ 화면에 주행 영상 띄우기)
            frame = env.render()
            frames.append(frame)

            if render:
                surf = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
                surf = pygame.transform.scale(surf, (1200, 800))
                screen.blit(surf, (0, 0))
                pygame.display.update()

            # 다음 스텝을 위해 prev_action은 신경망이 뱉은 원본 텐서를 줘야 함
            prev_h, prev_z, prev_action = h, z, action_tensor

            print(f"Action -> Steering: {env_action[0]:.2f}, Gas: {env_action[1]:.2f}, Brake: {env_action[2]:.2f} | Reward: {reward:.2f}")

            if terminated or truncated:
                break

    print(f"최종 점수: {total_eval_reward:.2f}")
    env.close()
    if render:
        pygame.quit()

    print("\n비디오 파일 생성 중...")
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    # ActionRepeat이 4이므로, 원본 60fps / 4 = 15fps가 실제 주행 속도
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"저장 완료! 파일 위치: {video_path}")
    return total_eval_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 Dreamer 에이전트 주행 평가")
    parser.add_argument("--ckpt", default="latest", help="반복 횟수, 'latest', 'best'")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--video", default="videos/eval_run.mp4", help="저장할 영상 경로 (.mp4 또는 .gif)")
    parser.add_argument("--no-render", action="store_true", help="pygame 창 없이 영상만 저장")
    args = parser.parse_args()
    evaluate(args.ckpt, args.output_dir, render=not args.no_render, video_path=args.video)
