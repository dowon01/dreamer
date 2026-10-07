import argparse
import os
import cv2
import imageio
import numpy as np
import torch
from models.world_model import WorldModel
from models.actor_critic import Actor
from utils.checkpoint import checkpoint_paths, resolve_tag
from utils.env import make_env, resize_obs, obs_to_tensor, to_env_action, REWARD_SCALE

# 월드 모델 시각화
#   recon   : 매 스텝 실제 화면을 보고 복원한 이미지 vs 실제
#   dream   : 실제와 같은 행동만 넣고 화면 없이 예측한 이미지 vs 실제 (resync 스텝마다 실제 화면으로 재동기화)
#   imagine : 화면 없이 actor가 상상 속에서 운전

def load_models(ckpt, output_dir, device):
    world_model = WorldModel().to(device)
    actor = Actor(latent_dim=world_model.latent_dim).to(device)
    paths = checkpoint_paths(output_dir, resolve_tag(output_dir, ckpt))
    world_model.load_state_dict(torch.load(paths['wm'], map_location=device))
    actor.load_state_dict(torch.load(paths['actor'], map_location=device))
    world_model.eval(); actor.eval()
    return world_model, actor

def decode(world_model, h, z):
    img = world_model.observation_decoder(torch.cat([h, z], dim=-1)).mean[0]
    return ((img.permute(1, 2, 0).cpu().numpy() + 0.5).clip(0, 1) * 255).astype(np.uint8)

def panel(img, title, sub, scale=5):
    img = cv2.resize(img, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)

def row(left, right):
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    return np.concatenate([left, gap, right], axis=1)

def drive(world_model, actor, device, steps):
    # 실제 주행하며 관측, 직전 행동, posterior 상태를 기록
    env = make_env()
    obs, _ = env.reset()
    obs = resize_obs(obs)
    observations, actions, states = [], [], []
    h, z = world_model.initial_state(1, device)
    prev_action = torch.zeros(1, 3, device=device)
    with torch.no_grad():
        for _ in range(steps):
            embed = world_model.encoder(obs_to_tensor(obs, device))
            h, z, _, _ = world_model.rssm(z, prev_action, h, embed)
            observations.append(obs); states.append((h, z)); actions.append(prev_action)
            action = actor(torch.cat([h, z], dim=-1), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(to_env_action(action.cpu().numpy()[0]))
            obs = resize_obs(obs)
            prev_action = action
            if terminated or truncated:
                break
    env.close()
    return observations, actions, states

def recon_video(world_model, actor, device, start, length):
    observations, _, states = drive(world_model, actor, device, start + length)
    frames, errors = [], []
    with torch.no_grad():
        for t in range(start, len(observations)):
            rec = decode(world_model, *states[t])
            err = np.abs(rec.astype(np.float32) - observations[t]).mean()
            errors.append(err)
            frames.append(row(panel(observations[t], "Real", f"t={t}"),
                              panel(rec, "Reconstruction (sees every frame)", f"t={t}  pixel err={err:5.1f}")))
    print(f"복원 평균 픽셀 오차: {np.mean(errors):.1f} / 255")
    return frames

def dream_video(world_model, actor, device, start, horizon, resync=15, context_show=10):
    observations, actions, states = drive(world_model, actor, device, start + horizon)
    horizon = min(horizon, len(observations) - start)
    frames = []
    with torch.no_grad():
        # 문맥 구간: 실제 화면을 보며 상태를 맞춤
        for t in range(start - context_show, start):
            frames.append(row(panel(observations[t], "Real", f"t={t}"),
                              panel(decode(world_model, *states[t]), "Dream", f"t={t}  observing real frames")))
        # 상상 구간: 화면 없이 실제와 같은 행동만 넣어 예측
        for k in range(horizon):
            t = start + k
            if k == 0 or (resync and k % resync == 0):
                h, z = states[t - 1]
            h, z, _, _ = world_model.rssm(z, actions[t], h, None)
            dreamed = k % resync + 1 if resync else k + 1
            err = np.abs(decode(world_model, h, z).astype(np.float32) - observations[t]).mean()
            frames.append(row(panel(observations[t], "Real", f"t={t}"),
                              panel(decode(world_model, h, z), "Dream (same actions, no frames)", f"{dreamed} steps without frames  err={err:4.1f}")))
    return frames

def imagine_video(world_model, actor, device, start, horizon, context_show=10):
    observations, _, states = drive(world_model, actor, device, start)
    frames = []
    with torch.no_grad():
        for t in range(start - context_show, start):
            frames.append(panel(observations[t], "Real (observing)", f"t={t}"))
        h, z = states[-1]
        total = 0.0
        for k in range(horizon):
            a = actor(torch.cat([h, z], dim=-1), deterministic=True)
            h, z, _, _ = world_model.rssm(z, a, h, None)
            total += world_model.predict_reward(torch.cat([h, z], -1)).mean.item() * REWARD_SCALE
            frames.append(panel(decode(world_model, h, z), "Imagination (actor drives)", f"+{k + 1} step  dreamed reward={total:6.1f}"))
    return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="월드 모델의 복원/상상 결과를 실제 주행과 비교하는 영상 생성")
    parser.add_argument("--mode", default="all", choices=["recon", "dream", "imagine", "all"])
    parser.add_argument("--ckpt", default="best", help="반복 횟수, 'latest', 'best'")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--start", type=int, default=60, help="영상 시작 스텝 (dream은 이 시점부터 화면 없이 상상)")
    parser.add_argument("--length", type=int, default=60, help="영상 길이(스텝)")
    parser.add_argument("--resync", type=int, default=15, help="dream: N스텝마다 실제 화면으로 재동기화 (0이면 끝까지 상상만)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model, actor = load_models(args.ckpt, args.output_dir, device)
    os.makedirs("videos", exist_ok=True)
    if args.mode in ("recon", "all"):
        imageio.mimsave("videos/dream_recon.gif", recon_video(world_model, actor, device, args.start, args.length), fps=10, loop=0)
        print("저장 완료! 파일 위치: videos/dream_recon.gif")
    if args.mode in ("dream", "all"):
        imageio.mimsave("videos/dream_compare.gif", dream_video(world_model, actor, device, args.start, args.length, args.resync), fps=10, loop=0)
        print("저장 완료! 파일 위치: videos/dream_compare.gif")
    if args.mode in ("imagine", "all"):
        imageio.mimsave("videos/dream_imagine.gif", imagine_video(world_model, actor, device, args.start, max(args.length, 100)), fps=10, loop=0)
        print("저장 완료! 파일 위치: videos/dream_imagine.gif")
