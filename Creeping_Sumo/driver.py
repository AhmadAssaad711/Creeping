"""
driver.py
----------
PPO training + test driver.

v2: Zone speed caps REMOVED. The reward function alone produces
    creeping behavior via speed-conditioned progress reward and
    quadratic approach penalty.  The agent's action is acceleration
    (bounded by physical limits a_min=-4, a_max=+2 m/s²), integrated
    to a target speed that is passed to SUMO uncapped.

    The only constraint on v_cmd is v_cmd >= 0 (can't go backward)
    and the physical accel/decel limits.  The agent must learn
    through reward shaping that arriving fast at J1 yields near-zero
    progress reward and accumulates approach penalties.

Curriculum:
    Phase 1 (0-15%):   300 veh/hr, neutral     — learn basic crossing + approach
    Phase 2 (15-40%):  500 veh/hr, conservative+neutral — learn yield response
    Phase 3 (40-100%): 700 veh/hr, full mix    — learn abort on aggressive
"""

import sys
import os
import re
import math
import random
import numpy as np
import logging
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from distutils.util import strtobool

from .Networks.agent import PPOAgent
from .simulation.environment import SumoEnv, OBS_DIM
from .simulation.settings import MAX_SPEED

# Zone speed caps REMOVED (v2).
# The reward function now handles speed shaping entirely via
# speed-conditioned progress and quadratic approach penalty.
# No external action-space constraint is needed.
# The agent's acceleration is still bounded by a_min/a_max (-4, +2),
# which IS a physical constraint (hardware braking/accel limits).


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
def _bool(s):
    if s not in {"False", "True"}:
        raise ValueError("Use True/False")
    return s == "True"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name",    type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=1)
    p.add_argument("--learning-rate",   type=float, default=3e-4)
    p.add_argument("--total-timesteps", type=int,   default=2_000_000)
    p.add_argument("--episode-length",  type=int,   default=600)
    p.add_argument("--train",          type=_bool, default=True)
    p.add_argument("--test-timesteps", type=int,   default=20_000)
    p.add_argument("--cfg",     type=str, default="Scenario3.sumocfg")
    p.add_argument("--nogui",   action="store_true")
    p.add_argument("--town",    type=str, default="Scenario3")
    p.add_argument("--delay-ms",type=int, default=0)
    p.add_argument("--step-length",         type=float, default=0.05)
    p.add_argument("--lateral-resolution",  type=float, default=0.02)
    p.add_argument("--ego",         type=str, default="ego")
    p.add_argument("--route",       type=str, default="-E2,-E0")
    p.add_argument("--departLane",  type=str, default="0")
    p.add_argument("--departPos",   type=str, default="0")
    p.add_argument("--departSpeed", type=str, default="0")
    p.add_argument("--curriculum", type=lambda x: bool(strtobool(x)),
                   default=True, nargs="?", const=True)
    p.add_argument("--load-checkpoint",  type=_bool, default=False)
    p.add_argument("--checkpoint-path",  type=str,   default=None)
    p.add_argument("--torch-deterministic",
                   type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    p.add_argument("--cuda",
                   type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    p.add_argument("--print-every", type=int, default=50)
    return p.parse_args()


# ──────────────────────────────────────────────
# Curriculum helpers
# ──────────────────────────────────────────────
def _curriculum_phase(timestep: int, total: int) -> int:
    frac = timestep / max(total, 1)
    if frac < 0.15:
        return 1
    elif frac < 0.40:
        return 2
    return 3


def _rou_path(cfg_path: str, phase: int) -> str:
    base = os.path.dirname(cfg_path)
    if phase == 1:
        return os.path.join(base, "Scenario3_phase1.rou.xml")
    elif phase == 2:
        return os.path.join(base, "Scenario3_phase2.rou.xml")
    return os.path.join(base, "Scenario3.rou.xml")


def _write_phase1(path: str) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="neutral"
           accel="2.5" decel="4.0" emergencyDecel="6.0"
           sigma="0.4" minGap="2.5" maxSpeed="13.89"
           speedFactor="1.0" speedDev="0.1"
           tau="1.2" impatience="0.2"
           carFollowModel="IDM" color="0,200,0"/>
    <flow id="f_0" begin="0" end="3600" perHour="300" from="E0"  to="E1"  type="neutral"/>
    <flow id="f_1" begin="0" end="3600" perHour="300" from="-E1" to="-E0" type="neutral"/>
</routes>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def _write_phase2(path: str) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vTypeDistribution id="phase2_dist">
        <vType id="conservative_p2" probability="0.5"
               accel="2.0" decel="4.5" emergencyDecel="6.0"
               sigma="0.3" minGap="3.0" maxSpeed="13.89"
               speedFactor="0.85" speedDev="0.05"
               tau="1.8" impatience="0.0"
               carFollowModel="IDM" color="0,0,200"/>
        <vType id="neutral_p2" probability="0.5"
               accel="2.5" decel="4.0" emergencyDecel="6.0"
               sigma="0.4" minGap="2.5" maxSpeed="13.89"
               speedFactor="1.0" speedDev="0.1"
               tau="1.2" impatience="0.2"
               carFollowModel="IDM" color="0,200,0"/>
    </vTypeDistribution>
    <flow id="f_0" begin="0" end="3600" perHour="500" from="E0"  to="E1"  type="phase2_dist"/>
    <flow id="f_1" begin="0" end="3600" perHour="500" from="-E1" to="-E0" type="phase2_dist"/>
</routes>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def _patch_sumocfg(cfg_path: str, rou_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(
        r'(<route-files\s+value=")[^"]*(")',
        rf'\g<1>{os.path.basename(rou_path)}\g<2>',
        content,
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[curriculum] cfg → {os.path.basename(rou_path)}", flush=True)


def _entropy_std(timestep: int, total: int,
                 std_start: float = 0.4, std_end: float = 0.1) -> float:
    frac = min(1.0, timestep / max(total, 1))
    return std_start + frac * (std_end - std_start)


def _cleanup_env(env):
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"[driver] args: {args}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"[driver] device: {device}", flush=True)

    logging.basicConfig(level=logging.WARNING)

    env    = None
    writer = None

    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(this_dir, args.cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"cfg not found: {cfg_path}")

        ego_route = [e.strip() for e in args.route.split(",") if e.strip()]

        if args.curriculum:
            p1 = _rou_path(cfg_path, 1)
            p2 = _rou_path(cfg_path, 2)
            if not os.path.exists(p1):
                _write_phase1(p1)
                print(f"[curriculum] wrote {p1}", flush=True)
            if not os.path.exists(p2):
                _write_phase2(p2)
                print(f"[curriculum] wrote {p2}", flush=True)

        env = SumoEnv(
            cfg_path=cfg_path,
            use_gui=not args.nogui,
            step_length=float(args.step_length),
            delay_ms=(int(args.delay_ms) if not args.nogui else 0),
            lateral_resolution=float(args.lateral_resolution),
            ego_id=args.ego,
            ego_route_edges=ego_route,
            ego_depart_lane=args.departLane,
            ego_depart_pos=args.departPos,
            ego_depart_speed=args.departSpeed,
            ego_speed_mode=0,
            ego_lane_change_mode=0,
            nearby_radius=80.0,
            collision_distance=2.5,
            max_ttc=10.0,
            creep_zone_dist=12.0,
            ttc_critical=2.0,
            timeout_steps_in_zone=600,
        )
        print(f"[driver] obs_dim={OBS_DIM}", flush=True)

        agent = PPOAgent(
            args.town, obs_dim=OBS_DIM, action_dim=1,
            a_min=-4.0, a_max=2.0,
            action_std_init=0.4,
        )
        try:
            agent.policy.to(device)
            agent.old_policy.to(device)
        except Exception:
            pass
        agent.optimizer = torch.optim.Adam(
            agent.policy.parameters(), lr=args.learning_rate
        )

        ckpt_dir   = os.path.join("ppo_checkpoints", args.town)
        final_dir  = os.path.join(ckpt_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "final_weights.pth")

        dt = float(args.step_length)

        # ══════════════════════════════════════
        # TRAIN
        # ══════════════════════════════════════
        if args.train:
            timestep    = 0
            episode     = 0
            scores      = []
            tb_step_seed = 0
            resumed      = False
            cur_phase    = 0

            if args.load_checkpoint:
                ckpt_p = args.checkpoint_path or final_path
                try:
                    blob = torch.load(ckpt_p, map_location=device)
                    agent.policy.load_state_dict(blob["policy"])
                    agent.old_policy.load_state_dict(blob["policy"])
                    prog = blob.get("meta", {}).get("progress", {})
                    timestep     = int(prog.get("timestep", 0))
                    episode      = int(prog.get("episode", 0))
                    scores       = list(prog.get("scores_tail", []))
                    tb_step_seed = timestep
                    resumed = True
                    print(f"[resume] loaded t={timestep}", flush=True)
                except FileNotFoundError:
                    print("[resume] no checkpoint, starting fresh", flush=True)

            run_tag = f"S3_v4_{args.total_timesteps}"
            logdir  = f"runs/{run_tag}/{args.town}"
            writer  = SummaryWriter(
                log_dir=logdir, flush_secs=10,
                purge_step=(tb_step_seed if resumed else None),
            )

            obs = env.reset()
            obs_vec = np.asarray(obs["state"], dtype=np.float32)
            v_cmd = 0.0

            print(f"[driver] training start t={timestep}", flush=True)

            while timestep < args.total_timesteps:
                episode += 1

                # ── Curriculum ─────────────────────────────────────
                if args.curriculum:
                    phase = _curriculum_phase(timestep, args.total_timesteps)
                    if phase != cur_phase:
                        cur_phase = phase
                        new_rou   = _rou_path(cfg_path, phase)
                        _patch_sumocfg(cfg_path, new_rou)
                        print(f"[curriculum] phase {phase} at t={timestep}", flush=True)
                        writer.add_text("curriculum", f"Phase {phase}", timestep)

                # ── Entropy decay ───────────────────────────────────
                new_std = _entropy_std(timestep, args.total_timesteps)
                agent.set_action_std(new_std)

                obs     = env.reset()
                obs_vec = np.asarray(obs["state"], dtype=np.float32)
                v_cmd   = 0.0

                ep_return = 0.0
                ep_step   = 0
                t_start   = datetime.now()

                r_parts = {k: 0.0 for k in
                           ["r_safety", "r_approach", "r_creep", "r_yield", "r_progress", "r_gap"]}
                ttc_min  = float("inf")
                creep_speeds = []

                for _ in range(args.episode_length):
                    accel_cmd = agent.get_action(obs_vec, train=True)
                    if not math.isfinite(accel_cmd):
                        accel_cmd = 0.0

                    # Integrate acceleration → speed
                    v_cmd = max(0.0, v_cmd + accel_cmd * dt)

                    # v2: no zone speed cap — reward handles speed shaping
                    obs_next, reward, done, info = env.step(v_cmd, accel_cmd=accel_cmd)

                    ttc = float(info.get("ttc", 10.))
                    ttc_min = min(ttc_min, ttc)

                    for k in r_parts:
                        r_parts[k] += float(info.get(k, 0.))

                    if info.get("in_creep") if hasattr(info, "get") else False:
                        pass
                    if isinstance(obs_next, dict) and obs_next.get("in_creep"):
                        ego_spd_now = float(obs_next["state"][1]) * MAX_SPEED
                        if ego_spd_now > 0.1:
                            creep_speeds.append(ego_spd_now)

                    obs_vec = np.asarray(obs_next["state"], dtype=np.float32) \
                        if obs_next is not None else obs_vec

                    agent.memory.rewards.append(float(reward))
                    agent.memory.dones.append(bool(done))

                    obs = obs_next
                    timestep  += 1
                    ep_step   += 1
                    ep_return += float(reward)

                    if timestep % args.print_every == 0:
                        sm = info.get("speed_match", 0.)
                        print(
                            f"[t={timestep}] ep={episode} step={ep_step} "
                            f"v={v_cmd:.2f} a={accel_cmd:+.2f} "
                            f"ttc={ttc:.2f} sm={sm:.2f} r={reward:+.3f} done={done}",
                            flush=True,
                        )

                    if timestep % 100_000 == 0:
                        agent.save()

                    if done or timestep >= args.total_timesteps:
                        break

                scores.append(ep_return)

                mean_creep = float(np.mean(creep_speeds)) if creep_speeds else 0.0
                print(
                    f"Ep {episode:5d} | T {timestep:9d} | "
                    f"R {ep_return:7.2f} | Avg10 {np.mean(scores[-10:]):7.2f} | "
                    f"minTTC {ttc_min:.2f} | creepSpd {mean_creep:.2f}m/s | "
                    f"goal={info.get('reached_goal')} timeout={info.get('timeout')} "
                    f"std={new_std:.3f}",
                    flush=True,
                )

                writer.add_scalar("train/episode_return",       ep_return, episode)
                writer.add_scalar("train/episode_length_steps", ep_step,   episode)
                writer.add_scalar("train/min_ttc",              ttc_min,   episode)
                writer.add_scalar("train/mean_creep_speed",     mean_creep, episode)
                writer.add_scalar("train/reached_goal", int(info.get("reached_goal", False)), episode)
                writer.add_scalar("train/timeout",      int(info.get("timeout", False)),      episode)
                writer.add_scalar("train/action_std",   new_std,                              episode)

                for k, v in r_parts.items():
                    writer.add_scalar(f"reward/{k}", v / max(ep_step, 1), episode)

                if episode % 10 == 0:
                    agent.learn()
                    agent.chkpt_save()

                writer.flush()

            payload = {
                "policy":    agent.old_policy.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "meta": {
                    "town": args.town, "obs_dim": OBS_DIM,
                    "saved_at": datetime.now().isoformat(),
                    "progress": {
                        "timestep": timestep, "episode": episode,
                        "scores_tail": scores[-200:],
                    },
                },
            }
            torch.save(payload, final_path)
            print(f"[FINAL] saved → {final_path}", flush=True)

        # ══════════════════════════════════════
        # TEST
        # ══════════════════════════════════════
        else:
            ckpt_path = args.checkpoint_path or final_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            blob = torch.load(ckpt_path, map_location=device)
            agent.policy.load_state_dict(blob["policy"])
            agent.old_policy.load_state_dict(blob["policy"])
            agent.policy.eval()
            agent.old_policy.eval()
            agent.set_action_std(0.05)
            print(f"[TEST] loaded {ckpt_path}", flush=True)

            timestep = 0
            scores   = []

            while timestep < args.test_timesteps:
                obs     = env.reset()
                obs_vec = np.asarray(obs["state"], dtype=np.float32)
                v_cmd   = 0.0
                ep_return = 0.0

                for _ in range(args.episode_length):
                    accel_cmd = agent.get_action(obs_vec, train=False)
                    if not math.isfinite(accel_cmd):
                        accel_cmd = 0.0
                    v_cmd = max(0.0, v_cmd + accel_cmd * dt)

                    obs_next, reward, done, info = env.step(v_cmd, accel_cmd=accel_cmd)
                    obs_vec = np.asarray(obs_next["state"], dtype=np.float32)
                    obs = obs_next
                    timestep  += 1
                    ep_return += reward
                    if done or timestep >= args.test_timesteps:
                        break

                scores.append(ep_return)
                print(
                    f"[TEST] ep={len(scores)} R={ep_return:.2f} "
                    f"Avg10={np.mean(scores[-10:]):.2f} "
                    f"goal={info.get('reached_goal')} coll={info.get('collided')}",
                    flush=True,
                )

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _cleanup_env(env)
        print("[driver] exit", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)