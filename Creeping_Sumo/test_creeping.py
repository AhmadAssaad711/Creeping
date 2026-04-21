"""
test_creeping.py
-----------------
Behavioral inspection for trained creeping policies.

Per-step output in the creep zone:
  phase    APPROACH / CREEP_ZONE / JUNCTION / CROSS_DONE
  action   APPROACH / CREEP / HOLD / COMMIT / ABORT
  speed    m/s
  d_J1     metres remaining to J1
  TTC      seconds
  gap      best upcoming gap (s)
  cv_acc   nearest crossing vehicle acceleration (negative = braking = yielding)

Usage (from inside Creeping_Sumo folder):
  python test_creeping.py \\
      --checkpoint "...ppo_checkpoints/Scenario3_v4/final/final_weights.pth" \\
      --cfg Scenario3.sumocfg \\
      --episodes 20 \\
      --traffic creep_trigger \\
      --delay 100
"""

import os
import sys
import math
import re
import argparse
import numpy as np
import torch

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("Set SUMO_HOME before running.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from Creeping_Sumo.Networks.agent import PPOAgent
from Creeping_Sumo.simulation.environment import SumoEnv, OBS_DIM
from Creeping_Sumo.simulation.settings import MAX_SPEED

# ──────────────────────────────────────────────
# Traffic file helpers
# ──────────────────────────────────────────────
_TRAFFIC_XML = {
    "conservative": """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="conservative"
           accel="2.0" decel="4.5" emergencyDecel="6.0"
           sigma="0.3" minGap="3.0" maxSpeed="13.89"
           speedFactor="0.85" speedDev="0.05"
           tau="1.8" impatience="0.0"
           carFollowModel="IDM" color="0,0,200"/>
    <flow id="f_0" begin="0" end="3600" perHour="700" from="E0"  to="E1"  type="conservative"/>
    <flow id="f_1" begin="0" end="3600" perHour="700" from="-E1" to="-E0" type="conservative"/>
</routes>
""",
    "neutral": """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="neutral"
           accel="2.5" decel="4.0" emergencyDecel="6.0"
           sigma="0.4" minGap="2.5" maxSpeed="13.89"
           speedFactor="1.0" speedDev="0.1"
           tau="1.2" impatience="0.2"
           carFollowModel="IDM" color="0,200,0"/>
    <flow id="f_0" begin="0" end="3600" perHour="700" from="E0"  to="E1"  type="neutral"/>
    <flow id="f_1" begin="0" end="3600" perHour="700" from="-E1" to="-E0" type="neutral"/>
</routes>
""",
    "aggressive": """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="aggressive"
           accel="3.5" decel="3.0" emergencyDecel="6.0"
           sigma="0.5" minGap="1.5" maxSpeed="13.89"
           speedFactor="1.15" speedDev="0.1"
           tau="0.8" impatience="0.8"
           carFollowModel="IDM" color="200,0,0"/>
    <flow id="f_0" begin="0" end="3600" perHour="700" from="E0"  to="E1"  type="aggressive"/>
    <flow id="f_1" begin="0" end="3600" perHour="700" from="-E1" to="-E0" type="aggressive"/>
</routes>
""",
}


def _setup_traffic(cfg_path: str, traffic: str) -> None:
    base = os.path.dirname(cfg_path)
    if traffic == "mix":
        rou = os.path.join(base, "Scenario3.rou.xml")
    elif traffic in ("moderate", "creep_trigger"):
        rou = os.path.join(base, f"Scenario3_test_{traffic}.rou.xml")
        if not os.path.exists(rou):
            raise FileNotFoundError(
                f"{rou} not found. Download it and place it in {base}/")
    else:
        rou = os.path.join(base, f"Scenario3_test_{traffic}.rou.xml")
        with open(rou, "w", encoding="utf-8") as f:
            f.write(_TRAFFIC_XML[traffic])

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(
        r'(<route-files\s+value=")[^"]*(")',
        rf'\g<1>{os.path.basename(rou)}\g<2>',
        content,
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[test] traffic={traffic}  rou={os.path.basename(rou)}")


# ──────────────────────────────────────────────
# Action classifier
# ──────────────────────────────────────────────
def _classify_action(
    ego_speed: float, prev_speed: float,
    dist_j1: float, ttc: float, prev_ttc: float,
    creep_zone_dist: float = 12.0,
    ttc_critical: float = 2.0,
    optimal_creep: float = 2.0,
) -> str:
    if not math.isfinite(dist_j1) or dist_j1 > creep_zone_dist:
        return "APPROACH"

    decelerating = ego_speed < prev_speed - 0.08
    accelerating = ego_speed > prev_speed + 0.08
    danger       = ttc < ttc_critical
    ttc_rising   = (math.isfinite(ttc) and math.isfinite(prev_ttc) and
                    ttc - prev_ttc > 0.5)

    if danger and (decelerating or ego_speed < 0.5):
        return "HOLD"
    if accelerating and ttc_rising and ttc >= ttc_critical:
        return "COMMIT"
    if decelerating and not danger and ttc < 3.0:
        return "ABORT"
    if 0.2 < ego_speed <= optimal_creep + 2.0:
        return "CREEP"
    return "CREEP"


def _phase_label(dist_j1: float, creep_zone: float = 12.0) -> str:
    if not math.isfinite(dist_j1):
        return "CROSS_DONE"
    if dist_j1 > creep_zone:
        return "APPROACH  "
    if dist_j1 > 0.3:
        return "CREEP_ZONE"
    return "JUNCTION  "


# ──────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────
def run_episode(
    env: SumoEnv, agent: PPOAgent,
    episode_num: int, dt: float, ep_length: int,
) -> dict:
    obs      = env.reset()
    obs_vec  = np.asarray(obs["state"], dtype=np.float32)
    v_cmd    = 0.0

    prev_speed  = 0.0
    prev_ttc    = 10.0
    prev_action = "APPROACH"

    action_counts = {k: 0 for k in ["APPROACH", "CREEP", "HOLD", "COMMIT", "ABORT"]}
    min_ttc        = float("inf")
    gaps_accepted  = []
    gaps_rejected  = []
    creep_speeds   = []
    yield_events   = 0
    total_reward   = 0.0

    print(f"\n{'='*90}")
    print(f"  EPISODE {episode_num}")
    print(f"{'='*90}")
    print(f"  {'step':>5}  {'phase':<10}  {'action':<7}  "
          f"{'speed':>6}  {'d_J1':>6}  {'TTC':>5}  {'gap':>7}  {'cv_acc':>6}  note")
    print(f"  {'-'*88}")

    for step in range(ep_length):
        accel_cmd = agent.get_action(obs_vec, train=False)
        if not math.isfinite(accel_cmd):
            accel_cmd = 0.0

        v_cmd = max(0.0, v_cmd + accel_cmd * dt)
        dist_j1_now = obs.get("dist_j1", float("inf")) if isinstance(obs, dict) else float("inf")

        obs_next, reward, done, info = env.step(v_cmd, accel_cmd=accel_cmd)
        total_reward += float(reward)

        ttc      = float(info.get("ttc", 10.))
        dist_j1  = float(info.get("dist_j1", float("inf")))
        best_gap = float(info.get("best_gap", 0.))
        ego_speed = v_cmd

        # nearest cv acceleration from obs
        cv1_accel = float(obs_next["state"][7]) * 4.0 if obs_next is not None else 0.0

        min_ttc = min(min_ttc, ttc)
        if info.get("in_creep") or (obs_next and obs_next.get("in_creep")):
            if ego_speed > 0.1:
                creep_speeds.append(ego_speed)
        if cv1_accel < -1.5 and (info.get("in_creep") or dist_j1 <= 12.0):
            yield_events += 1

        action = _classify_action(ego_speed, prev_speed, dist_j1, ttc, prev_ttc)
        action_counts[action] = action_counts.get(action, 0) + 1

        if prev_action in ("CREEP", "HOLD") and action == "COMMIT":
            if math.isfinite(best_gap):
                gaps_accepted.append(round(best_gap, 2))
        if prev_action == "COMMIT" and action == "ABORT":
            if math.isfinite(best_gap):
                gaps_rejected.append(round(best_gap, 2))

        note = ""
        if action != prev_action:
            if action == "HOLD":
                note = f"<-- holding  TTC={ttc:.2f}s"
            elif action == "COMMIT":
                note = f"<-- commit   gap={best_gap:.1f}s"
            elif action == "ABORT":
                note = "<-- ABORT"
            elif action == "CREEP" and prev_action in ("HOLD", "APPROACH"):
                note = "<-- creeping"
        if cv1_accel < -2.0 and dist_j1 <= 12.0:
            note += f"  [YIELD acc={cv1_accel:.1f}]"

        in_zone = math.isfinite(dist_j1) and dist_j1 <= 12.0
        if in_zone or step % 5 == 0:
            ttc_s  = f"{ttc:5.2f}" if ttc < 9.9 else "  inf"
            gap_s  = f"{best_gap:6.2f}s" if math.isfinite(best_gap) and best_gap < 99 else "  inf  "
            acc_s  = f"{cv1_accel:+.2f}"
            print(
                f"  {step:5d}  {_phase_label(dist_j1)}  {action:<7}  "
                f"{ego_speed:5.2f}ms  {dist_j1:5.1f}m  {ttc_s}  {gap_s}  {acc_s}  {note}"
            )

        prev_speed  = ego_speed
        prev_ttc    = ttc
        prev_action = action

        if obs_next is not None:
            obs_vec = np.asarray(obs_next["state"], dtype=np.float32)
        obs = obs_next

        if done:
            break

    outcome = (
        "GOAL"      if info.get("reached_goal") else
        "COLLISION" if info.get("collided")     else
        "TIMEOUT"
    )

    mean_creep = float(np.mean(creep_speeds)) if creep_speeds else 0.0
    ac = action_counts

    print(f"\n  EPISODE {episode_num} SUMMARY")
    print(f"  Outcome              : {outcome}")
    print(f"  Steps                : {step+1}")
    print(f"  Total reward         : {total_reward:.2f}")
    print(f"  Min TTC              : {min_ttc:.2f}s")
    print(f"  Mean creep speed     : {mean_creep:.2f} m/s  (target: 2.0)")
    print(f"  Yield events         : {yield_events}  (crossing vehicle braked for ego)")
    print(f"  CREEP steps          : {ac.get('CREEP', 0)}")
    print(f"  HOLD  steps          : {ac.get('HOLD',  0)}")
    print(f"  ABORT steps          : {ac.get('ABORT', 0)}")
    print(f"  COMMIT steps         : {ac.get('COMMIT',0)}")
    print(f"  Gaps accepted        : {gaps_accepted}")
    print(f"  Gaps rejected        : {gaps_rejected}")

    return {
        "episode": episode_num, "outcome": outcome, "steps": step+1,
        "total_reward": total_reward, "min_ttc": min_ttc,
        "action_counts": ac, "mean_creep_speed": mean_creep,
        "yield_events": yield_events,
        "gaps_accepted": gaps_accepted, "gaps_rejected": gaps_rejected,
    }


def _final_summary(summaries: list, traffic: str) -> None:
    n   = len(summaries)
    oc  = [s["outcome"] for s in summaries]
    goals      = oc.count("GOAL")
    collisions = oc.count("COLLISION")
    timeouts   = oc.count("TIMEOUT")

    creep_steps   = [s["action_counts"].get("CREEP",  0) for s in summaries]
    hold_steps    = [s["action_counts"].get("HOLD",   0) for s in summaries]
    abort_steps   = [s["action_counts"].get("ABORT",  0) for s in summaries]
    commit_steps  = [s["action_counts"].get("COMMIT", 0) for s in summaries]
    min_ttcs      = [s["min_ttc"] for s in summaries if math.isfinite(s["min_ttc"])]
    creep_speeds  = [s["mean_creep_speed"] for s in summaries if s["mean_creep_speed"] > 0]
    yield_events  = [s["yield_events"] for s in summaries]
    all_accepted  = [g for s in summaries for g in s["gaps_accepted"]]
    all_rejected  = [g for s in summaries for g in s["gaps_rejected"]]
    rewards       = [s["total_reward"] for s in summaries]

    print(f"\n{'='*90}")
    print(f"  FINAL SUMMARY  |  {n} episodes  |  traffic: {traffic}")
    print(f"{'='*90}")
    print(f"  Success rate   : {goals}/{n}  ({100*goals/n:.1f}%)")
    print(f"  Collision rate : {collisions}/{n}  ({100*collisions/n:.1f}%)")
    print(f"  Timeout rate   : {timeouts}/{n}  ({100*timeouts/n:.1f}%)")
    print(f"  Avg reward/ep  : {np.mean(rewards):.2f}")
    print()
    print(f"  Avg CREEP  steps/ep : {np.mean(creep_steps):.1f}")
    print(f"  Avg HOLD   steps/ep : {np.mean(hold_steps):.1f}")
    print(f"  Avg ABORT  steps/ep : {np.mean(abort_steps):.1f}")
    print(f"  Avg COMMIT steps/ep : {np.mean(commit_steps):.1f}")
    if creep_speeds:
        print(f"  Mean creep speed    : {np.mean(creep_speeds):.2f} m/s  (target ~2.0)")
    if yield_events:
        print(f"  Avg yield events/ep : {np.mean(yield_events):.1f}")
    print()
    if min_ttcs:
        print(f"  Min TTC mean  : {np.mean(min_ttcs):.2f}s")
        print(f"  Min TTC worst : {np.min(min_ttcs):.2f}s")
    if all_accepted:
        print(f"  Accepted gaps : mean={np.mean(all_accepted):.2f}s  "
              f"min={np.min(all_accepted):.2f}s")
    if all_rejected:
        print(f"  Rejected gaps : mean={np.mean(all_rejected):.2f}s")

    print(f"\n  BEHAVIORAL CHECKS")
    checks = {
        "Creeping observed (>5 steps/ep avg)":
            np.mean(creep_steps) > 5,
        "HOLD observed (stops for danger)":
            np.mean(hold_steps) > 0,
        "ABORT observed (retreats from bad gaps)":
            np.mean(abort_steps) > 0,
        "Yield events observed (negotiation worked)":
            np.mean(yield_events) > 0,
        "Safety floor (worst TTC > 0.5s)":
            (min(min_ttcs) > 0.5) if min_ttcs else False,
        "Creep speed near optimal (1-3 m/s avg)":
            (1.0 <= np.mean(creep_speeds) <= 4.0) if creep_speeds else None,
        "Reasonable accepted gap (<8s avg)":
            (np.mean(all_accepted) < 8.0) if all_accepted else None,
    }
    for desc, result in checks.items():
        tag = "N/A" if result is None else ("PASS" if result else "FAIL")
        print(f"  {tag:<5} {desc}")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--cfg",            type=str, required=True)
    p.add_argument("--episodes",       type=int, default=20)
    p.add_argument("--episode-length", type=int, default=600)
    p.add_argument("--delay",          type=int, default=100)
    p.add_argument("--nogui",          action="store_true")
    p.add_argument("--traffic",        type=str, default="mix",
                   choices=["conservative", "neutral", "aggressive",
                            "mix", "moderate", "creep_trigger"])
    p.add_argument("--step-length",    type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path  = os.path.abspath(args.cfg)
    ckpt_path = os.path.abspath(args.checkpoint)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    _setup_traffic(cfg_path, args.traffic)

    env = SumoEnv(
        cfg_path=cfg_path,
        use_gui=not args.nogui,
        step_length=float(args.step_length),
        delay_ms=(int(args.delay) if not args.nogui else 0),
        lateral_resolution=0.02,
        ego_id="ego",
        ego_route_edges=["-E2", "-E0"],
        nearby_radius=80.0,
        collision_distance=2.5,
        max_ttc=10.0,
        creep_zone_dist=12.0,
        ttc_critical=2.0,
        timeout_steps_in_zone=600,
    )

    device  = torch.device("cpu")
    blob    = torch.load(ckpt_path, map_location=device)
    agent   = PPOAgent("test", obs_dim=OBS_DIM, action_dim=1, a_min=-4.0, a_max=2.0)
    agent.policy.load_state_dict(blob["policy"])
    agent.old_policy.load_state_dict(blob["policy"])
    agent.policy.eval()
    agent.old_policy.eval()
    agent.set_action_std(0.05)

    print(f"[test] checkpoint : {ckpt_path}")
    print(f"[test] obs_dim    : {OBS_DIM}")
    print(f"[test] traffic    : {args.traffic}")
    print(f"[test] episodes   : {args.episodes}")

    dt        = float(args.step_length)
    summaries = []

    try:
        for ep in range(1, args.episodes + 1):
            s = run_episode(env, agent, ep, dt, args.episode_length)
            summaries.append(s)
    finally:
        env.close()

    _final_summary(summaries, args.traffic)


if __name__ == "__main__":
    main()