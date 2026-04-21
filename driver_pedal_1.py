import sys
import random
import numpy as np
import logging
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from distutils.util import strtobool

# ---- your PPOAgent (same as your CARLA driver uses) ----
from .Networks.agent import PPOAgent  # adjust if your project structure differs

# ---- your SUMO env (this is the file where SumoEnv lives) ----
from .simulation.environment import SumoEnv  # adjust filename if needed


def boolean_string(s: str) -> bool:
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string (use True/False)")
    return s == "True"


def parse_args():
    parser = argparse.ArgumentParser()

    # bookkeeping (mirrors your driver) :contentReference[oaicite:1]{index=1}
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--env-name", type=str, default="sumo")

    # PPO/training
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--episode-length", type=int, default=1000)

    # run mode
    parser.add_argument("--train", type=boolean_string, default=True)
    parser.add_argument("--test-timesteps", type=int, default=20000)

    # SUMO/scenario
    parser.add_argument("--cfg", type=str, default="Scenario1.sumocfg")
    parser.add_argument("--nogui", action="store_true", help="Run with sumo (no GUI)")
    parser.add_argument("--town", type=str, default="Scenario1")  # kept for ckpt folder naming

    # ego spawn / route
    parser.add_argument("--ego", type=str, default="ego")
    parser.add_argument("--route", type=str, default="-E2,-E0", help='Comma-separated edges, e.g. "-E2,-E0"')
    parser.add_argument("--departLane", type=str, default="0")
    parser.add_argument("--departPos", type=str, default="0")
    parser.add_argument("--departSpeed", type=str, default="0")

    # env timing
    parser.add_argument("--step-length", type=float, default=0.05)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--lateral-resolution", type=float, default=0.1)

    # resume / checkpoints
    parser.add_argument("--load-checkpoint", type=boolean_string, default=False)
    parser.add_argument("--checkpoint-path", type=str, default=None)

    # system (mirrors your driver) :contentReference[oaicite:2]{index=2}
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True)

    # debug prints
    parser.add_argument("--print-every", type=int, default=25)

    return parser.parse_args()


def _build_optimizer(agent, lr: float):
    # same pattern as your driver: optimize policy parameters safely :contentReference[oaicite:3]{index=3}
    pol = getattr(agent, "policy", None)
    if pol is None:
        raise RuntimeError("PPOAgent has no .policy attribute")
    return torch.optim.Adam(pol.parameters(), lr=lr)


def _best_effort_env_cleanup(env):
    if env is None:
        return
    try:
        if hasattr(env, "close") and callable(env.close):
            env.close()
    except Exception:
        pass


def main():
    args = parse_args()
    train = args.train
    town = args.town
    total_timesteps = int(args.total_timesteps)
    episode_length = int(args.episode_length)

    print(f"[driver-sumo] args: {args}")

    # Reproducibility (same as your driver) :contentReference[oaicite:4]{index=4}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[driver-sumo] device: {device}")

    logging.basicConfig(level=logging.INFO)

    env = None
    writer = None

    try:
        # ---------------- ENV INIT ----------------
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = args.cfg
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(this_dir, cfg_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find SUMO config at: {cfg_path}")

        ego_route_edges = [e.strip() for e in args.route.split(",") if e.strip()]

        print("[driver-sumo] before env init", flush=True)
        env = SumoEnv(
            cfg_path=cfg_path,
            use_gui=not args.nogui,
            step_length=float(args.step_length),
            delay_ms=(int(args.delay_ms) if not args.nogui else 0),
            lateral_resolution=float(args.lateral_resolution),
            ego_id=args.ego,
            nearby_radius=80.0,
            collision_distance=2.5,
            max_ttc=20.0,
            ego_route_edges=ego_route_edges,
            ego_depart_lane=args.departLane,
            ego_depart_pos=args.departPos,
            ego_depart_speed=args.departSpeed,
            ego_speed_mode=0,
            ego_lane_change_mode=0,
            K_closest=1,
        )
        print("[driver-sumo] after env init", flush=True)

        # ---------------- AGENT INIT ----------------
        # We infer obs_dim from env.reset() once (since you said you’ll match dims later).
        obs = env.reset()
        if "state" not in obs:
            raise RuntimeError("Env obs must include obs['state'] (fixed numeric vector) for PPO.")

        obs_dim = int(np.asarray(obs["state"], dtype=np.float32).shape[0])
        agent = PPOAgent(town, obs_dim=obs_dim, action_dim=1, a_min=-3.0, a_max=2.0)

        # move nets to device (same spirit as your driver) :contentReference[oaicite:5]{index=5}
        try:
            agent.policy.to(device)
            if hasattr(agent, "old_policy"):
                agent.old_policy.to(device)
        except Exception:
            pass

        agent.optimizer = _build_optimizer(agent, lr=args.learning_rate)

        # ---------------- CHECKPOINT PATHS ----------------
        ckpt_dir = os.path.join("ppo_checkpoints", town)
        final_dir = os.path.join(ckpt_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "final_weights.pth")

        # ---------------- TRAIN ----------------
        if train:
            timestep = 0
            episode = 0
            scores = []
            ep_len_seconds = []
            tb_step_seed = 0
            resumed = False

            if args.load_checkpoint:
                try:
                    if args.checkpoint_path:
                        path, meta = agent.load(args.checkpoint_path)
                    else:
                        path, meta = agent.load()
                    print(f"[resume] Loaded: {path}")
                    prog = (meta or {}).get("progress", {})
                    timestep = int(prog.get("timestep", 0))
                    episode = int(prog.get("episode", 0))
                    scores = list(prog.get("scores_tail", []))
                    ep_len_seconds = list(prog.get("ep_len_seconds_tail", []))
                    tb_step_seed = int(prog.get("tb_step", timestep))
                    resumed = True
                except FileNotFoundError:
                    print("[resume] No checkpoint found; starting fresh.")

            run_tag = f"SUMO_TM_{total_timesteps}"
            logdir = f"runs/{run_tag}/{town}"
            writer = SummaryWriter(
                log_dir=logdir,
                flush_secs=2,
                filename_suffix=f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                purge_step=(tb_step_seed if resumed else None),
            )

            print(f"[driver-sumo] starting train while-loop, timestep = {timestep}", flush=True)

            dt = float(env.step_length)

            while timestep < total_timesteps:
                episode += 1

                
                obs = env.reset()
                obs_vec = np.asarray(obs["state"], dtype=np.float32)

                # per-episode aggregates
                ttc_sum, ttc_min, ttc_count = 0.0, float("inf"), 0

                ep_return = 0.0
                ep_step = 0
                t_start = datetime.now()

                # driver-side speed command (because PPO outputs accel; env.step expects target speed)
                evx, evy = obs["ego_vel"]
                v_cmd = float((evx * evx + evy * evy) ** 0.5)

                for _ in range(episode_length):
                    # IMPORTANT: call get_action ONCE per step (same as your driver) :contentReference[oaicite:6]{index=6}
                    raw_action = agent.get_action(obs_vec, train=True)

                    # scalar accel
                    try:
                        accel_cmd = float(np.asarray(raw_action).item())
                    except Exception:
                        accel_cmd = float(raw_action)

                    if not np.isfinite(accel_cmd):
                        accel_cmd = 0.0

                    # integrate accel -> target speed
                    v_cmd = max(0.0, v_cmd + accel_cmd * dt)

                    obs_next, reward, done, info = env.step(v_cmd)

                    if isinstance(info, dict) and "ttc" in info:
                        t = float(info["ttc"])
                        ttc_sum += t
                        ttc_min = min(ttc_min, t)
                        ttc_count += 1

                    # next obs
                    if obs_next is None:
                        done = True
                    else:
                        obs_vec = np.asarray(obs_next["state"], dtype=np.float32)

                    # PPO memory: rewards/dones only (states/actions/logprobs already stored by get_action)
                    agent.memory.rewards.append(float(reward))
                    agent.memory.dones.append(bool(done))

                    timestep += 1
                    ep_step += 1
                    ep_return += float(reward)

                    if (timestep % max(args.print_every, 1)) == 0:
                        print(
                            f"[step] t={timestep} ep={episode} ep_step={ep_step} "
                            f"a={accel_cmd:+.2f} v_cmd={v_cmd:.2f} "
                            f"ttc={float(info.get('ttc', 0.0)):.2f} r={float(reward):+.3f} done={done}",
                            flush=True,
                        )

                    # periodic save
                    if timestep % 50000 == 0:
                        agent.save()

                    if done or timestep >= total_timesteps:
                        break

                scores.append(ep_return)
                ep_secs = (datetime.now() - t_start).total_seconds()
                ep_len_seconds.append(ep_secs)

                info_print = info if isinstance(info, dict) else {}
                print(
                    f"Episode {episode:5d} | Timestep {timestep:9d} | "
                    f"Reward {ep_return:8.2f} | Avg {np.mean(scores[-10:]):8.2f} | info={info_print}",
                    flush=True,
                )
                # ----- Episode-level TensorBoard logging -----
                writer.add_scalar("train/episode_return", float(ep_return), episode)
                writer.add_scalar("train/episode_length_steps", int(ep_step), episode)
                writer.add_scalar("train/episode_length_seconds", float(ep_secs), episode)
                writer.add_scalar("train/avg_return_last10", float(np.mean(scores[-10:])), episode)

                if ttc_count > 0:
                    writer.add_scalar("train/episode_mean_ttc", ttc_sum / ttc_count, episode)
                    writer.add_scalar("train/episode_min_ttc", ttc_min, episode)

                # Save progress for resume (same idea) :contentReference[oaicite:8]{index=8}
                last10 = scores[-10:] if len(scores) >= 10 else scores
                last100 = scores[-100:] if len(scores) >= 100 else scores
                agent.runtime_progress = {
                    "timestep": timestep,
                    "episode": episode,
                    "last_reward": float(ep_return),
                    "avg_last10": float(np.mean(last10)) if last10 else 0.0,
                    "avg_last100": float(np.mean(last100)) if last100 else 0.0,
                    "scores_tail": scores[-200:],
                    "ep_len_seconds_tail": ep_len_seconds[-200:],
                    "tb_step": timestep,
                }

                # PPO update every 10 episodes (same as your driver) :contentReference[oaicite:9]{index=9}
                if episode % 10 == 0:
                    agent.learn()
                    agent.chkpt_save()

                writer.flush()

            # Save final weights payload
            payload = {
                "policy": agent.old_policy.state_dict() if hasattr(agent, "old_policy") else agent.policy.state_dict(),
                "optimizer": agent.optimizer.state_dict() if agent.optimizer is not None else None,
                "meta": {
                    "town": town,
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_timesteps": total_timesteps,
                    "run_tag": run_tag,
                },
            }
            torch.save(payload, final_path)
            print(f"[FINAL] Saved final weights → {final_path}", flush=True)

        # ---------------- TEST ----------------
        else:
            # Load trained weights (final or provided path)
            ckpt_path = args.checkpoint_path or os.path.join("ppo_checkpoints", town, "final", "final_weights.pth")
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(os.getcwd(), ckpt_path)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"[TEST] checkpoint not found: {ckpt_path}")

            payload = torch.load(ckpt_path, map_location=device)
            agent.policy.load_state_dict(payload["policy"])
            if hasattr(agent, "old_policy"):
                agent.old_policy.load_state_dict(payload["policy"])
            agent.policy.eval()
            if hasattr(agent, "old_policy"):
                agent.old_policy.eval()

            print(f"[TEST] Loaded policy from: {ckpt_path}", flush=True)
            timestep = 0
            episode = 0
            scores = []

            dt = float(env.step_length)

            while timestep < int(args.test_timesteps):
                episode += 1
                obs = env.reset()
                obs_vec = np.asarray(obs["state"], dtype=np.float32)

                evx, evy = obs["ego_vel"]
                v_cmd = float((evx * evx + evy * evy) ** 0.5)

                ep_return = 0.0

                for _ in range(int(args.episode_length)):
                    raw_action = agent.get_action(obs_vec, train=False)
                    try:
                        accel_cmd = float(np.asarray(raw_action).item())
                    except Exception:
                        accel_cmd = float(raw_action)

                    if not np.isfinite(accel_cmd):
                        accel_cmd = 0.0

                    v_cmd = max(0.0, v_cmd + accel_cmd * dt)

                    obs_next, reward, done, info = env.step(v_cmd)
                    if obs_next is None:
                        done = True
                    else:
                        obs_vec = np.asarray(obs_next["state"], dtype=np.float32)

                    timestep += 1
                    ep_return += float(reward)

                    if done or timestep >= int(args.test_timesteps):
                        break

                scores.append(ep_return)
                print(
                    f"[TEST] Episode {episode:5d} | Timestep {timestep:9d} | "
                    f"Reward {ep_return:8.2f} | Avg {np.mean(scores[-10:]):8.2f}",
                    flush=True,
                )

    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        _best_effort_env_cleanup(env)
        print("\nExit")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()