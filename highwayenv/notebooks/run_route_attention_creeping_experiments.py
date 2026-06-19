from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import (  # noqa: E402
    evaluate_route_agent,
    evaluate_route_reference_policies,
    load_route_ppo,
    route_config_for,
    train_route_ppo,
)


MIN_INITIAL_VEHICLES = 12
MIN_SPAWN_PROBABILITY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run route-aware HighwayEnv creeping PPO experiments.")
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/results/intersectionRouteAttentionCreeping"))
    parser.add_argument("--maneuvers", nargs="+", default=["straight", "left"], choices=["straight", "left"])
    parser.add_argument("--initial-vehicles", type=int, default=MIN_INITIAL_VEHICLES)
    parser.add_argument("--spawn-probability", type=float, default=MIN_SPAWN_PROBABILITY)
    parser.add_argument("--duration", type=int, default=22)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--backend", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--reference-episodes", type=int, default=50)
    parser.add_argument("--standard-steps", type=int, default=40000)
    parser.add_argument("--creeping-steps", type=int, default=60000)
    parser.add_argument("--attention-steps", type=int, default=90000)
    parser.add_argument("--agents", nargs="+", default=["standard", "reward", "attention"], choices=["standard", "reward", "attention"])
    parser.add_argument("--tag", default="", help="Optional suffix for agent/model names, e.g. conflict_ttc.")
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.initial_vehicles < MIN_INITIAL_VEHICLES:
        parser.error(f"--initial-vehicles must be at least {MIN_INITIAL_VEHICLES} for this benchmark")
    if args.spawn_probability < MIN_SPAWN_PROBABILITY:
        parser.error(f"--spawn-probability must be at least {MIN_SPAWN_PROBABILITY} for this benchmark")
    return args


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "models").mkdir(exist_ok=True)
    config_path = args.results_dir / "experiment_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    all_episode_frames: list[pd.DataFrame] = []
    all_summary_frames: list[pd.DataFrame] = []

    for maneuver_index, maneuver in enumerate(args.maneuvers):
        cfg = route_config_for(
            maneuver,
            initial_vehicle_count=args.initial_vehicles,
            spawn_probability=args.spawn_probability,
            duration=args.duration,
        )
        print(f"\n=== {maneuver.upper()} references ===", flush=True)
        ref_df, ref_summary = evaluate_route_reference_policies(
            cfg,
            episodes=args.reference_episodes,
            seed=args.seed + 1000 * maneuver_index,
        )
        ref_df.to_csv(args.results_dir / f"{maneuver}_reference_eval_episodes.csv", index=False)
        ref_summary.to_csv(args.results_dir / f"{maneuver}_reference_summary.csv", index=False)
        print(ref_summary.to_string(index=False), flush=True)
        all_episode_frames.append(ref_df)
        all_summary_frames.append(ref_summary)

        tag = f"_{args.tag}" if args.tag else ""
        candidate_specs = {
            "standard": ("ppo_standard" + tag, "standard", False, args.standard_steps, args.seed + 101 + 1000 * maneuver_index),
            "reward": ("ppo_reward_creeping" + tag, "creeping", False, args.creeping_steps, args.seed + 202 + 1000 * maneuver_index),
            "attention": (
                "ppo_attention_reward_creeping" + tag,
                "creeping",
                True,
                args.attention_steps,
                args.seed + 303 + 1000 * maneuver_index,
            ),
        }
        specs = [candidate_specs[name] for name in args.agents]
        for agent_name, variant, use_attention_obs, timesteps, seed in specs:
            model_path = args.results_dir / "models" / f"{maneuver}_{agent_name}_{timesteps}.zip"
            print(f"\n=== {maneuver.upper()} {agent_name} train/eval ===", flush=True)
            start = time.time()
            if model_path.exists() and not args.overwrite:
                print(f"Loading existing model: {model_path}", flush=True)
                model = load_route_ppo(model_path, device=args.device)
            else:
                model = train_route_ppo(
                    variant=variant,
                    cfg=cfg,
                    total_timesteps=timesteps,
                    model_path=model_path,
                    n_envs=args.n_envs,
                    seed=seed,
                    device=args.device,
                    backend=args.backend,
                    use_attention_obs=use_attention_obs,
                )
            train_eval_s = time.time() - start
            eval_df, summary = evaluate_route_agent(
                model,
                agent_name=agent_name,
                variant=variant,
                cfg=cfg,
                episodes=args.eval_episodes,
                seed=args.seed + 5000 + 1000 * maneuver_index,
                use_attention_obs=use_attention_obs,
            )
            eval_df["model_path"] = str(model_path)
            summary["model_path"] = str(model_path)
            summary["train_eval_wall_time_s"] = train_eval_s
            eval_df.to_csv(args.results_dir / f"{maneuver}_{agent_name}_eval_episodes.csv", index=False)
            summary.to_csv(args.results_dir / f"{maneuver}_{agent_name}_summary.csv", index=False)
            print(summary.to_string(index=False), flush=True)
            all_episode_frames.append(eval_df)
            all_summary_frames.append(summary)

    combined_eval = pd.concat(all_episode_frames, ignore_index=True)
    combined_summary = pd.concat(all_summary_frames, ignore_index=True)
    combined_eval.to_csv(args.results_dir / "combined_eval_episodes.csv", index=False)
    combined_summary.to_csv(args.results_dir / "combined_summary.csv", index=False)
    print("\n=== COMBINED SUMMARY ===", flush=True)
    print(combined_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
