from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import make_route_intersection_env, route_config_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether dense intersection resets contain collisions that are "
            "unavoidable by ego-only control. The ego speed is forced to zero "
            "immediately after reset, then the slowest discrete action is stepped."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=150000)
    parser.add_argument("--maneuvers", nargs="+", choices=["straight", "left"], default=["straight", "left"])
    parser.add_argument("--initial-vehicles", type=int, default=12)
    parser.add_argument("--spawn-probability", type=float, default=0.5)
    parser.add_argument("--duration", type=int, default=22)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/results/intersectionRouteAttentionCreeping"))
    return parser.parse_args()


def force_ego_stationary(env) -> dict:
    ego = getattr(env.unwrapped, "vehicle", None)
    if ego is None:
        return {"ego_found": False, "initial_ego_speed": math.nan}

    initial_speed = float(getattr(ego, "speed", math.nan))
    ego.speed = 0.0
    if hasattr(ego, "velocity"):
        try:
            ego.velocity = np.zeros_like(np.asarray(ego.velocity, dtype=float))
        except AttributeError:
            pass
    if hasattr(ego, "target_speed"):
        ego.target_speed = 0.0
    if hasattr(ego, "action"):
        try:
            ego.action["acceleration"] = min(float(ego.action.get("acceleration", 0.0)), -10.0)
        except Exception:
            pass
    return {"ego_found": True, "initial_ego_speed": initial_speed}


def audit_one_reset(maneuver: str, seed: int, args: argparse.Namespace) -> dict:
    cfg = route_config_for(
        maneuver,
        initial_vehicle_count=args.initial_vehicles,
        spawn_probability=args.spawn_probability,
        duration=args.duration,
        negotiating_traffic=False,
    )
    env = make_route_intersection_env("creeping", cfg=cfg, seed=seed, use_attention_obs=False)
    obs, _ = env.reset(seed=seed)
    force_info = force_ego_stationary(env)

    collided = False
    arrived = False
    final_info = {}
    steps_taken = 0
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(0)
        final_info = info
        steps_taken += 1
        collided = collided or bool(info.get("crashed", False))
        arrived = arrived or bool(info.get("arrived", False))
        if terminated or truncated:
            break

    ego = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    ego_speed = float(getattr(ego, "speed", math.nan)) if ego is not None else math.nan
    ego_position = np.asarray(getattr(ego, "position", [math.nan, math.nan]), dtype=float) if ego is not None else np.array([math.nan, math.nan])
    vehicle_count = len(getattr(road, "vehicles", [])) if road is not None else 0
    env.close()

    return {
        "maneuver": maneuver,
        "seed": seed,
        "forced_zero_steps": steps_taken,
        "collided_after_forced_zero": bool(collided),
        "arrived_after_forced_zero": bool(arrived),
        "ego_found": bool(force_info["ego_found"]),
        "initial_ego_speed": force_info["initial_ego_speed"],
        "ego_speed_after_step": ego_speed,
        "ego_distance_from_center": float(np.linalg.norm(ego_position)) if np.isfinite(ego_position).all() else math.nan,
        "vehicle_count_after_reset": vehicle_count,
        "min_ttc_seen": final_info.get("min_ttc_seen", math.nan),
        "min_conflict_ttc_seen": final_info.get("min_conflict_ttc_seen", math.nan),
        "initial_vehicle_count": args.initial_vehicles,
        "spawn_probability": args.spawn_probability,
        "duration": args.duration,
        "negotiating_traffic": False,
        "surrounding_vehicle_control": "unmodified_highway_env_IDMVehicle",
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for maneuver_index, maneuver in enumerate(args.maneuvers):
        base_seed = args.seed + 10_000 * maneuver_index
        for episode in range(args.episodes):
            row = audit_one_reset(maneuver, base_seed + episode, args)
            row["episode"] = episode
            rows.append(row)

    df = pd.DataFrame(rows)
    tag = f"forced_zero_ego_reset_audit_{args.initial_vehicles}_{str(args.spawn_probability).replace('.', '_')}_{args.episodes}eps"
    episode_path = args.results_dir / f"{tag}.csv"
    df.to_csv(episode_path, index=False)

    summary = (
        df.groupby("maneuver", dropna=False)
        .agg(
            episodes=("episode", "count"),
            forced_zero_collision_rate=("collided_after_forced_zero", "mean"),
            forced_zero_collisions=("collided_after_forced_zero", "sum"),
            mean_initial_ego_speed=("initial_ego_speed", "mean"),
            mean_vehicle_count_after_reset=("vehicle_count_after_reset", "mean"),
        )
        .reset_index()
    )
    summary["initial_vehicle_count"] = args.initial_vehicles
    summary["spawn_probability"] = args.spawn_probability
    summary["duration"] = args.duration
    summary["forced_zero_steps"] = args.steps
    summary["negotiating_traffic"] = False
    summary["surrounding_vehicle_control"] = "unmodified_highway_env_IDMVehicle"
    summary_path = args.results_dir / f"{tag}_summary.csv"
    summary.to_csv(summary_path, index=False)

    failures = df[df["collided_after_forced_zero"].astype(bool)].copy()
    failures_path = args.results_dir / f"{tag}_failures.csv"
    failures.to_csv(failures_path, index=False)

    print(summary.to_string(index=False))
    print(f"episodes: {episode_path}")
    print(f"summary: {summary_path}")
    print(f"failures: {failures_path}")


if __name__ == "__main__":
    main()
