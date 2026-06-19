from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import (  # noqa: E402
    RouteFastPolicy,
    evaluate_route_agent,
    load_route_ppo,
    route_config_for,
)


MIN_INITIAL_VEHICLES = 12
MIN_SPAWN_PROBABILITY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one route-intersection episode and emit JSON.")
    parser.add_argument("--maneuver", choices=["straight", "left"], required=True)
    parser.add_argument(
        "--policy",
        choices=[
            "learned_creeping",
            "fast_non_creeping",
            "learned_safety",
            "learned_safety_progress",
            "learned_safety_timegate",
            "trained_safety_interaction_v3",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
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


def main() -> None:
    args = parse_args()
    cfg_kwargs = {
        "initial_vehicle_count": args.initial_vehicles,
        "spawn_probability": args.spawn_probability,
        "duration": args.duration,
        "negotiating_traffic": False,
    }
    variant = "creeping"
    use_attention_obs = True

    if args.policy in {
        "learned_safety",
        "learned_safety_progress",
        "learned_safety_timegate",
        "trained_safety_interaction_v3",
    }:
        cfg_kwargs.update(
            {
                "safety_shield": True,
                "safety_ellipse_margin": 0.15,
                "safety_conflict_radius": 18.0,
                "safety_prediction_horizon": 3.0,
                "safety_near_field_radius": 12.0,
                "safety_near_field_horizon": 0.6,
                "safety_time_gate": False,
                "safety_min_clearance": 0.0,
                "safety_emergency_decel": 7.0,
                "interaction_reward_focus": True,
            }
        )
    if args.policy == "learned_safety_timegate":
        cfg_kwargs.update(
            {
                "safety_ellipse_margin": 0.10,
                "safety_conflict_radius": 14.0,
                "safety_prediction_horizon": 2.5,
                "safety_time_gate": True,
                "safety_time_gate_buffer": 0.2,
                "safety_progress_assist": False,
            }
        )
    if args.policy == "learned_safety_progress":
        cfg_kwargs.update(
            {
                "safety_progress_assist": True,
                "safety_progress_assist_radius": 55.0,
            }
        )
    cfg = route_config_for(args.maneuver, **cfg_kwargs)

    if args.policy == "fast_non_creeping":
        model = RouteFastPolicy()
        variant = "standard"
    elif args.policy == "trained_safety_interaction_v3":
        model_path = args.results_dir / "models" / f"{args.maneuver}_ppo_attention_safety12_05_interaction_v3_5000.zip"
        model = load_route_ppo(model_path, device="cpu")
    else:
        model_path = args.results_dir / "models" / f"{args.maneuver}_ppo_attention_bc_creeping_calibrated_v2.zip"
        model = load_route_ppo(model_path, device="cpu")

    tag = f"benchmark{args.initial_vehicles}_{str(args.spawn_probability).replace('.', '_')}"
    df, _ = evaluate_route_agent(
        model,
        agent_name=f"{tag}_{args.maneuver}_{args.policy}",
        variant=variant,
        cfg=cfg,
        episodes=1,
        seed=args.seed,
        deterministic=True,
        use_attention_obs=use_attention_obs,
    )
    row = df.iloc[0].to_dict()
    row.update(
        {
            "policy_family": args.policy,
            "initial_vehicle_count": args.initial_vehicles,
            "spawn_probability": args.spawn_probability,
            "duration": args.duration,
            "negotiating_traffic": False,
            "surrounding_vehicle_control": "unmodified_highway_env_IDMVehicle",
            "simulator_crash": False,
            "worker_seed": args.seed,
        }
    )
    for key, value in list(row.items()):
        if isinstance(value, (np.bool_, bool)):
            row[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            row[key] = int(value)
        elif isinstance(value, (np.floating,)):
            row[key] = None if not np.isfinite(value) else float(value)
        elif isinstance(value, float) and not np.isfinite(value):
            row[key] = None
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
