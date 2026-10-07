import argparse
import logging
import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from utils.buffer import ReplayBuffer, save_episode
from utils.checkpoint import checkpoint_paths, resolve_tag
from utils.env import make_env, resize_obs, obs_to_tensor, to_env_action, sample_random_action, REWARD_SCALE
from utils.train_world_model import train_world_model
from utils.train_actor_critic import train_actor_critic, ReturnNormalizer

def shaping_penalty(action, prev_steer):
    # 조향각 미세 조정 -> 핸들을 계속 꺾는 현상을 줄여, 자연스러운 주행을 위해
    current_steer = action[0] # 현재 조향 (-1 ~ 1)
    current_gas = action[1]   # 현재 엑셀 (-1 ~ 1, 실제론 0 이상일때 가속)

    # 핸들을 확 꺾으면 감점 (이전 각도와의 차이)
    steer_penalty = 0.1 * abs(current_steer - prev_steer)

    # 브레이크 유도: 핸들을 크게 꺾었는데 엑셀도 밟고 있으면 감점
    corner_penalty = 0.05 * abs(current_steer) * max(0, current_gas)

    return steer_penalty + corner_penalty

def run_episode(env, device, policy=None, data_dir=None, prefix="episode", shaping=False):
    # policy=None 이면 랜덤 행동, data_dir가 주어지면 에피소드를 저장
    # 반환값의 total_reward는 페널티가 없는 실제 환경 보상
    obs, _ = env.reset()
    obs = resize_obs(obs)

    episode_obs, episode_act, episode_rew, episode_done = [obs], [np.zeros(3, np.float32)], [0.0], [0.0]

    if policy is not None:
        world_model, actor, deterministic = policy
        prev_h, prev_z = world_model.initial_state(1, device)
        prev_action = torch.zeros(1, 3, device=device)

    total_reward = 0.0
    step_count = 0
    prev_steer = 0.0 # 조향각 변수 저장
    done = False

    while not done:
        if policy is None:
            action = sample_random_action()
        else:
            with torch.no_grad():
                embed = world_model.encoder(obs_to_tensor(obs, device))
                h, z, _, _ = world_model.rssm(prev_z, prev_action, prev_h, embed)
                action_tensor = actor(torch.cat([h, z], dim=-1), deterministic=deterministic)
            action = action_tensor.cpu().numpy()[0]
            prev_h, prev_z, prev_action = h, z, action_tensor

        # 환경 한 스텝 진행
        next_obs, reward, terminated, truncated, _ = env.step(to_env_action(action))
        done = terminated or truncated
        total_reward += reward

        if shaping: # 훈련 중에만 페널티를 주어 습관을 고침
            reward -= shaping_penalty(action, prev_steer)
            prev_steer = action[0]

        obs = resize_obs(next_obs)
        episode_obs.append(obs)
        episode_act.append(action)
        episode_rew.append(reward / REWARD_SCALE) # 보상 스케일링
        # 시간 제한(truncated)은 실패가 아니므로 종료로 학습하지 않음
        episode_done.append(float(terminated))
        step_count += 1

    if data_dir is not None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_episode(os.path.join(data_dir, f"{prefix}_{timestamp}.npz"),
                     episode_obs, episode_act, episode_rew, episode_done)

    return total_reward, step_count

def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def parse_args():
    parser = argparse.ArgumentParser(description="Dreamer MBRL for CarRacing-v3")
    parser.add_argument("--resume", default=None, help="재개할 체크포인트 (반복 횟수, 'latest', 'best'). 생략하면 처음부터 학습")
    parser.add_argument("--iterations", type=int, default=10000, help="마지막 반복 번호")
    parser.add_argument("--data-dir", default="data/mbrl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed-episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=500, help="버퍼에 유지할 최대 에피소드 수")
    parser.add_argument("--wm-lr", type=float, default=2e-4)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--keep-last", type=int, default=5, help="이번 실행에서 저장한 체크포인트 중 유지할 개수")
    parser.add_argument("--no-shaping", action="store_true", help="조향 페널티(보상 쉐이핑) 끄기")
    parser.add_argument("--log-dir", default="logs", help="반복마다 한 줄씩 요약을 남길 로그 폴더")
    return parser.parse_args()

def setup_logger(log_dir, run_name):
    # tqdm 진행바와 섞이지 않도록 반복 요약만 파일에 기록
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logger = logging.getLogger("mbrl")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    def log_exception(exc_type, exc, tb):
        logger.error("학습 중단", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = log_exception
    return logger, log_path

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()

    run_name = f"dreamer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    print("TensorBoard 시작: tensorboard --logdir runs")
    logger, log_path = setup_logger(args.log_dir, run_name)
    print(f"학습 로그: {log_path}")
    logger.info(f"start args={vars(args)} device={device}")

    world_model = WorldModel().to(device)
    actor = Actor(latent_dim=world_model.latent_dim).to(device)
    critic = Critic(latent_dim=world_model.latent_dim).to(device)
    target_critic = Critic(latent_dim=world_model.latent_dim).to(device)

    wm_opt = torch.optim.Adam(world_model.parameters(), lr=args.wm_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    return_norm = ReturnNormalizer()

    start_iter, global_step, best_eval = 1, 0, -float("inf")
    if args.resume is None:
        world_model.apply(initialize_weights)
        actor.apply(initialize_weights)
        critic.apply(initialize_weights)
        target_critic.load_state_dict(critic.state_dict())
    else:
        tag = resolve_tag(args.output_dir, args.resume)
        paths = checkpoint_paths(args.output_dir, tag)
        world_model.load_state_dict(torch.load(paths['wm'], map_location=device))
        actor.load_state_dict(torch.load(paths['actor'], map_location=device))
        critic.load_state_dict(torch.load(paths['critic'], map_location=device))
        target_critic.load_state_dict(critic.state_dict())
        if os.path.exists(paths['train_state']):
            state = torch.load(paths['train_state'], map_location=device)
            wm_opt.load_state_dict(state['wm_opt'])
            actor_opt.load_state_dict(state['actor_opt'])
            critic_opt.load_state_dict(state['critic_opt'])
            target_critic.load_state_dict(state['target_critic'])
            if 'return_norm' in state:
                return_norm.load_state_dict(state['return_norm'])
            # 옵티마이저 상태에 저장된 lr 대신 이번 실행 인자의 lr을 사용
            for opt, lr in ((wm_opt, args.wm_lr), (actor_opt, args.actor_lr), (critic_opt, args.critic_lr)):
                for g in opt.param_groups: g['lr'] = lr
            start_iter = state['iteration'] + 1
            global_step = state['global_step']
            best_eval = state.get('best_eval', best_eval)
        elif isinstance(tag, int):
            start_iter = tag + 1
        print(f"체크포인트 로드: {paths['wm']} (iteration {start_iter}부터 재개)")
        logger.info(f"resume from {paths['wm']} start_iter={start_iter} global_step={global_step}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    num_existing = len([f for f in os.listdir(args.data_dir) if f.endswith('.npz')])
    if num_existing < args.seed_episodes:
        print("시드 데이터 수집 중...")
        for _ in tqdm(range(args.seed_episodes - num_existing), desc="Random Seed"):
            run_episode(env, device, policy=None, data_dir=args.data_dir, prefix="seed")

    buffer = ReplayBuffer(args.data_dir, seq_len=args.seq_len, batch_size=args.batch_size, max_episodes=args.max_episodes)

    def save_checkpoint(tag, iteration):
        paths = checkpoint_paths(args.output_dir, tag)
        torch.save(actor.state_dict(), paths['actor'])
        torch.save(world_model.state_dict(), paths['wm'])
        torch.save(critic.state_dict(), paths['critic'])
        torch.save({
            'wm_opt': wm_opt.state_dict(),
            'actor_opt': actor_opt.state_dict(),
            'critic_opt': critic_opt.state_dict(),
            'target_critic': target_critic.state_dict(),
            'return_norm': return_norm.state_dict(),
            'iteration': iteration,
            'global_step': global_step,
            'best_eval': best_eval,
        }, paths['train_state'])

    saved_iters = [] # 이번 실행에서 저장한 체크포인트 (오래된 것부터 삭제)

    print("\nMBRL start")

    for iteration in range(start_iter, args.iterations + 1):
        print(f"\n=== Iteration {iteration} ===")
        iter_start = time.time()

        # 1. 수집
        train_reward, step_count = run_episode(env, device, policy=(world_model, actor, False),
                                               data_dir=args.data_dir, shaping=not args.no_shaping)
        buffer.load_new_data()
        writer.add_scalar("Rollout/Train_Reward", train_reward, iteration)
        writer.add_scalar("Rollout/Train_Step_Count", step_count, iteration)
        print(f"[훈련 보상] {train_reward:.2f}")

        # 2. Update-To-Data(생존 스텟 비례 학습)
        num_updates = int(step_count / 2) # 살아남은 프레임 수의 절반에 비례하여 업데이트 횟수 결정
        pbar = tqdm(range(num_updates), desc="   Training (WM & AC)", leave=False)
        loss_sums = {}
        for step in pbar:
            batch = buffer.sample_batch()

            # World Model 학습
            wm_loss, detached_hs, detached_zs = train_world_model(world_model, wm_opt, batch, device, is_train=True)

            # Actor-Critic 학습
            ac_loss = train_actor_critic(world_model, actor, critic, target_critic, actor_opt, critic_opt, detached_hs, detached_zs, device, return_norm)

            global_step += 1
            for k, v in {**{f"wm_{k}": v for k, v in wm_loss.items()}, **ac_loss}.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

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

        # 3. 평가 (노이즈 없이 여러 에피소드 평균)
        eval_reward = None
        if iteration % args.eval_every == 0:
            eval_rewards = [run_episode(env, device, policy=(world_model, actor, True))[0]
                            for _ in range(args.eval_episodes)]
            eval_reward = float(np.mean(eval_rewards))
            writer.add_scalar("Rollout/Eval_Reward", eval_reward, iteration)
            print(f"[실전 평가 보상] {eval_reward:.2f} (노이즈 제거, {args.eval_episodes}회 평균)")
            if eval_reward > best_eval:
                best_eval = eval_reward
                save_checkpoint("best", iteration)
                print(f"최고 성능 갱신 -> {args.output_dir}/*_best.pth")

        # 4. 모델 저장 및 시각화
        if iteration % args.save_every == 0:
            save_checkpoint(iteration, iteration)
            saved_iters.append(iteration)
            while len(saved_iters) > args.keep_last:
                for p in checkpoint_paths(args.output_dir, saved_iters.pop(0)).values():
                    if os.path.exists(p): os.remove(p)

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
                recon_img = (recon + 0.5).clamp(0, 1)

                grid = torch.cat([orig_img, recon_img], dim=-1)
                writer.add_images("Visual/Real_vs_Recon", grid[:4], iteration)

            print("모델 저장 & 시각화 완료")

        # 5. 반복 요약 로그 (한 줄)
        loss_means = {k: v / max(num_updates, 1) for k, v in loss_sums.items()}
        eval_str = f" eval={eval_reward:.1f} best={best_eval:.1f}" if eval_reward is not None else ""
        logger.info(f"iter={iteration} train={train_reward:.1f} steps={step_count}{eval_str} "
                    + " ".join(f"{k}={v:.4f}" for k, v in loss_means.items())
                    + f" gstep={global_step} sec={time.time() - iter_start:.0f}")

    logger.info("finished")
    env.close()
    writer.close()
