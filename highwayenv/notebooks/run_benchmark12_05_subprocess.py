from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import summarize_route_metrics  # noqa: E402


MIN_INITIAL_VEHICLES = 12
MIN_SPAWN_PROBABILITY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-tolerant 12/0.5 dense benchmark.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=140000)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["learned_creeping", "fast_non_creeping", "learned_safety"],
        choices=[
            "learned_creeping",
            "fast_non_creeping",
            "learned_safety",
            "learned_safety_progress",
            "learned_safety_timegate",
            "trained_safety_interaction_v3",
        ],
    )
    parser.add_argument("--maneuvers", nargs="+", default=["straight", "left"], choices=["straight", "left"])
    parser.add_argument("--initial-vehicles", type=int, default=MIN_INITIAL_VEHICLES)
    parser.add_argument("--spawn-probability", type=float, default=MIN_SPAWN_PROBABILITY)
    parser.add_argument("--duration", type=int, default=22)
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/results/intersectionRouteAttentionCreeping"))
    args = parser.parse_args()
    if args.initial_vehicles < MIN_INITIAL_VEHICLES:
        parser.error(f"--initial-vehicles must be at least {MIN_INITIAL_VEHICLES} for this benchmark")
    if args.spawn_probability < MIN_SPAWN_PROBABILITY:
        parser.error(f"--spawn-probability must be at least {MIN_SPAWN_PROBABILITY} for this benchmark")
    return args


def crash_row(maneuver: str, policy: str, episode: int, seed: int, returncode: int, stderr: str) -> dict:
    return {
        "maneuver": maneuver,
        "agent": f"benchmark12_05_{maneuver}_{policy}",
        "episode": episode,
        "return": 0.0,
        "steps": 0,
        "survival_time_s": 0.0,
        "collided": True,
        "collision_count": 1,
        "success": False,
        "timeout": False,
        "time_to_collision_s": 0.0,
        "policy_family": policy,
        "initial_vehicle_count": None,
        "spawn_probability": None,
        "duration": None,
        "negotiating_traffic": False,
        "surrounding_vehicle_control": "unmodified_highway_env_IDMVehicle",
        "simulator_crash": True,
        "worker_seed": seed,
        "worker_returncode": returncode,
        "worker_stderr": stderr[-2000:],
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parent / "route_episode_worker.py"
    all_rows: list[dict] = []
    all_summaries: list[pd.DataFrame] = []

    for maneuver_index, maneuver in enumerate(args.maneuvers):
        for policy_index, policy in enumerate(args.policies):
            rows: list[dict] = []
            base_seed = args.seed + 10_000 * maneuver_index + 1_000 * policy_index
            print(f"=== {maneuver} {policy} ===", flush=True)
            for episode in range(args.episodes):
                seed = base_seed + episode
                cmd = [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    str(worker),
                    "--maneuver",
                    maneuver,
                    "--policy",
                    policy,
                    "--seed",
                    str(seed),
                    "--initial-vehicles",
                    str(args.initial_vehicles),
                    "--spawn-probability",
                    str(args.spawn_probability),
                    "--duration",
                    str(args.duration),
                    "--results-dir",
                    str(args.results_dir),
                ]
                completed = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, timeout=90)
                if completed.returncode == 0:
                    try:
                        row = json.loads(completed.stdout.strip().splitlines()[-1])
                    except Exception as exc:
                        row = crash_row(maneuver, policy, episode, seed, completed.returncode, f"{exc}\n{completed.stderr}")
                else:
                    row = crash_row(maneuver, policy, episode, seed, completed.returncode, completed.stderr)
                row["episode"] = episode
                row["initial_vehicle_count"] = args.initial_vehicles
                row["spawn_probability"] = args.spawn_probability
                row["duration"] = args.duration
                rows.append(row)
                all_rows.append(row)
                if (episode + 1) % 10 == 0:
                    successes = sum(bool(r.get("success", False)) for r in rows)
                    collisions = sum(bool(r.get("collided", False)) for r in rows)
                    crashes = sum(bool(r.get("simulator_crash", False)) for r in rows)
                    print(
                        f"{episode + 1:03d}/{args.episodes}: success={successes} collision={collisions} simulator_crash={crashes}",
                        flush=True,
                    )

            df = pd.DataFrame(rows)
            tag = f"benchmark{args.initial_vehicles}_{str(args.spawn_probability).replace('.', '_')}"
            episode_path = args.results_dir / f"{tag}_{maneuver}_{policy}_{args.episodes}eps_subprocess_eval_episodes.csv"
            df.to_csv(episode_path, index=False)
            summary = summarize_route_metrics(df)
            summary["policy_family"] = policy
            summary["initial_vehicle_count"] = args.initial_vehicles
            summary["spawn_probability"] = args.spawn_probability
            summary["duration"] = args.duration
            summary["negotiating_traffic"] = False
            summary["surrounding_vehicle_control"] = "unmodified_highway_env_IDMVehicle"
            summary["simulator_crash_rate"] = df["simulator_crash"].astype(bool).mean()
            summary_path = args.results_dir / f"{tag}_{maneuver}_{policy}_{args.episodes}eps_subprocess_summary.csv"
            summary.to_csv(summary_path, index=False)
            all_summaries.append(summary)
            print(summary.to_string(index=False), flush=True)

    all_eval = pd.DataFrame(all_rows)
    tag = f"benchmark{args.initial_vehicles}_{str(args.spawn_probability).replace('.', '_')}"
    all_eval.to_csv(args.results_dir / f"{tag}_all_{args.episodes}eps_subprocess_eval_episodes.csv", index=False)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.results_dir / f"{tag}_all_{args.episodes}eps_subprocess_summary.csv", index=False)
    print("=== COMBINED ===", flush=True)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
