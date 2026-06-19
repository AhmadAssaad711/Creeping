from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import (  # noqa: E402
    evaluate_route_agent,
    finetune_route_ppo,
    load_route_ppo,
    route_config_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune calibrated creeping policies inside the 12/0.5 safety-shielded interaction dynamics."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/results/intersectionRouteAttentionCreeping"))
    parser.add_argument("--maneuvers", nargs="+", choices=["straight", "left"], default=["straight", "left"])
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=160000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--backend", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safety_cfg(maneuver: str):
    return route_config_for(
        maneuver,
        initial_vehicle_count=12,
        spawn_probability=0.5,
        duration=22,
        negotiating_traffic=False,
        safety_shield=True,
        safety_ellipse_margin=0.15,
        safety_conflict_radius=18.0,
        safety_prediction_horizon=3.0,
        safety_near_field_radius=12.0,
        safety_near_field_horizon=0.6,
        safety_time_gate=False,
        safety_min_clearance=0.0,
        safety_emergency_decel=7.0,
        interaction_reward_focus=True,
    )


def main() -> None:
    args = parse_args()
    models_dir = args.results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_eval = []
    all_summary = []
    for maneuver_index, maneuver in enumerate(args.maneuvers):
        cfg = safety_cfg(maneuver)
        base_model = models_dir / f"{maneuver}_ppo_attention_bc_creeping_calibrated_v2.zip"
        model_path = models_dir / f"{maneuver}_ppo_attention_safety12_05_interaction_v3_{args.timesteps}.zip"
        if not base_model.exists():
            raise FileNotFoundError(base_model)

        start = time.time()
        if model_path.exists() and not args.overwrite:
            model = load_route_ppo(model_path, device=args.device)
            trained = False
        else:
            model = finetune_route_ppo(
                base_model_path=base_model,
                variant="creeping",
                cfg=cfg,
                total_timesteps=args.timesteps,
                model_path=model_path,
                n_envs=args.n_envs,
                seed=args.seed + 1000 * maneuver_index,
                device=args.device,
                backend=args.backend,
                use_attention_obs=True,
                learning_rate=args.learning_rate,
            )
            trained = True
        wall_time_s = time.time() - start

        eval_df, summary = evaluate_route_agent(
            model,
            agent_name=f"shielded_interaction_v3_{maneuver}_{args.timesteps}",
            variant="creeping",
            cfg=cfg,
            episodes=args.eval_episodes,
            seed=args.seed + 5000 + 1000 * maneuver_index,
            deterministic=True,
            use_attention_obs=True,
        )
        for frame in [eval_df, summary]:
            frame["policy_family"] = "trained_safety_interaction_v3"
            frame["model_path"] = str(model_path)
            frame["initial_vehicle_count"] = 12
            frame["spawn_probability"] = 0.5
            frame["duration"] = 22
            frame["negotiating_traffic"] = False
            frame["surrounding_vehicle_control"] = "unmodified_highway_env_IDMVehicle"
            frame["timesteps"] = args.timesteps
            frame["trained_this_run"] = trained
            frame["train_or_load_wall_time_s"] = wall_time_s
        all_eval.append(eval_df)
        all_summary.append(summary)
        eval_df.to_csv(args.results_dir / f"shielded_interaction_v3_{maneuver}_{args.timesteps}_eval_episodes.csv", index=False)
        summary.to_csv(args.results_dir / f"shielded_interaction_v3_{maneuver}_{args.timesteps}_summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)

    combined_eval = pd.concat(all_eval, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_eval.to_csv(args.results_dir / f"shielded_interaction_v3_{args.timesteps}_combined_eval_episodes.csv", index=False)
    combined_summary.to_csv(args.results_dir / f"shielded_interaction_v3_{args.timesteps}_combined_summary.csv", index=False)
    print("=== COMBINED ===", flush=True)
    print(combined_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
