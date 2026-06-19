from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd
from highway_env import utils as highway_utils
from highway_env.envs import intersection_env as highway_intersection_env
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork

sys.path.insert(0, str(Path(__file__).resolve().parent))

from highway_route_attention_creeping_utils import (  # noqa: E402
    ego_route_state,
    has_arrived,
    route_min_conflict_ttc,
    route_min_ttc,
    summarize_route_metrics,
)


def patch_regulated_road_conflict_prediction() -> None:
    """Avoid a Windows crash in route-based conflict prediction during dense left turns."""

    def safe_is_conflict_possible(v1, v2, horizon: int = 3, step: float = 0.25) -> bool:
        times = np.arange(step, horizon, step)
        v1_heading = float(getattr(v1, "heading", 0.0))
        v2_heading = float(getattr(v2, "heading", 0.0))
        v1_velocity = float(getattr(v1, "speed", 0.0)) * np.array([np.cos(v1_heading), np.sin(v1_heading)])
        v2_velocity = float(getattr(v2, "speed", 0.0)) * np.array([np.cos(v2_heading), np.sin(v2_heading)])
        v1_position = np.asarray(getattr(v1, "position", np.zeros(2)), dtype=float)
        v2_position = np.asarray(getattr(v2, "position", np.zeros(2)), dtype=float)

        for t in times:
            position_1 = v1_position + v1_velocity * t
            position_2 = v2_position + v2_velocity * t
            if np.linalg.norm(position_2 - position_1) > max(v1.LENGTH, v2.LENGTH):
                continue
            if highway_utils.rotated_rectangles_intersect(
                (position_1, 1.5 * v1.LENGTH, 0.9 * v1.WIDTH, v1_heading),
                (position_2, 1.5 * v2.LENGTH, 0.9 * v2.WIDTH, v2_heading),
            ):
                return True
        return False

    RegulatedRoad.is_conflict_possible = staticmethod(safe_is_conflict_possible)


def patch_intersection_road_class(road_class: str) -> None:
    if road_class == "plain":
        highway_intersection_env.RegulatedRoad = Road
    elif road_class == "safe_regulated":
        patch_regulated_road_conflict_prediction()
    elif road_class == "native":
        return
    else:
        raise ValueError(f"Unknown road class mode: {road_class}")


def patch_closest_lane_lookup() -> None:
    def approximate_lane_distance(lane, position: np.ndarray) -> float:
        if hasattr(lane, "start") and hasattr(lane, "end"):
            start = np.asarray(lane.start, dtype=float)
            end = np.asarray(lane.end, dtype=float)
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom <= 1e-9:
                return float(np.linalg.norm(position - start))
            t = float(np.clip(np.dot(position - start, segment) / denom, 0.0, 1.0))
            closest = start + t * segment
            return float(np.linalg.norm(position - closest))
        if hasattr(lane, "center") and hasattr(lane, "radius"):
            center = np.asarray(lane.center, dtype=float)
            return float(abs(np.linalg.norm(position - center) - float(lane.radius)))
        return float("inf")

    def safe_get_closest_lane_index(self, position, heading=None):
        indexes, distances = [], []
        position = np.asarray(position, dtype=float)
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, lane in enumerate(lanes):
                    distance = approximate_lane_distance(lane, position)
                    if math.isfinite(distance):
                        distances.append(distance)
                        indexes.append((_from, _to, _id))
        if not indexes:
            raise ValueError("No valid lane distance found")
        return indexes[int(np.argmin(distances))]

    RoadNetwork.get_closest_lane_index = safe_get_closest_lane_index


@dataclass(frozen=True)
class NativeIntersectionConfig:
    maneuver: str
    destination: str
    initial_vehicle_count: int = 12
    spawn_probability: float = 0.5
    duration: int = 22
    policy_frequency: int = 5
    simulation_frequency: int = 15
    vehicles_count: int = 15
    reward_speed_low: float = 7.0
    reward_speed_high: float = 9.0


def config_for(maneuver: str, args: argparse.Namespace) -> NativeIntersectionConfig:
    destination = "o2" if maneuver == "straight" else "o1"
    return NativeIntersectionConfig(
        maneuver=maneuver,
        destination=destination,
        initial_vehicle_count=args.initial_vehicles,
        spawn_probability=args.spawn_probability,
        duration=args.duration,
    )


def native_env_config(cfg: NativeIntersectionConfig) -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": cfg.vehicles_count,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": False,
            "observe_intentions": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9],
        },
        "policy_frequency": cfg.policy_frequency,
        "simulation_frequency": cfg.simulation_frequency,
        "duration": cfg.duration,
        "destination": cfg.destination,
        "controlled_vehicles": 1,
        "initial_vehicle_count": cfg.initial_vehicle_count,
        "spawn_probability": cfg.spawn_probability,
        "collision_reward": -5,
        "high_speed_reward": 1,
        "arrived_reward": 1,
        "reward_speed_range": [cfg.reward_speed_low, cfg.reward_speed_high],
        "normalize_reward": False,
        "offroad_terminal": False,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 7.15,
        "render_agent": True,
        "offscreen_rendering": False,
    }


def make_native_env(cfg: NativeIntersectionConfig, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = gym.make("intersection-v0", render_mode=render_mode)
    env.unwrapped.configure(native_env_config(cfg))
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(cfg: NativeIntersectionConfig, n_envs: int, seed: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def thunk(rank: int):
        def _init():
            return make_native_env(cfg, seed=seed + rank)

        return _init

    return VecMonitor(DummyVecEnv([thunk(rank) for rank in range(n_envs)]))


def train_ppo(cfg: NativeIntersectionConfig, model_path: Path, total_timesteps: int, n_envs: int, seed: int, device: str):
    from stable_baselines3 import PPO

    env = make_vec_env(cfg, n_envs=n_envs, seed=seed)
    n_steps = 128
    rollout_size = n_steps * n_envs
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=min(512, rollout_size),
        n_epochs=4,
        gamma=0.985,
        gae_lambda=0.92,
        learning_rate=2.5e-4,
        ent_coef=0.015,
        clip_range=0.2,
        seed=seed,
        device=device,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def resume_train_ppo(
    cfg: NativeIntersectionConfig,
    model_path: Path,
    total_timesteps: int,
    n_envs: int,
    seed: int,
    device: str,
):
    from stable_baselines3 import PPO

    env = make_vec_env(cfg, n_envs=n_envs, seed=seed)
    if model_path.exists():
        model = PPO.load(model_path, env=env, device=device)
        reset_num_timesteps = False
    else:
        n_steps = 128
        rollout_size = n_steps * n_envs
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=min(512, rollout_size),
            n_epochs=4,
            gamma=0.985,
            gae_lambda=0.92,
            learning_rate=2.5e-4,
            ent_coef=0.015,
            clip_range=0.2,
            seed=seed,
            device=device,
            verbose=1,
        )
        reset_num_timesteps = True
    model.learn(total_timesteps=total_timesteps, progress_bar=False, reset_num_timesteps=reset_num_timesteps)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def evaluate_ppo(model, cfg: NativeIntersectionConfig, episodes: int, seed: int, agent_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for episode in range(episodes):
        env = make_native_env(cfg, seed=seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0
        total_reward = 0.0
        collided = False
        arrived = False
        min_ttc_seen = math.inf
        min_conflict_ttc_seen = math.inf
        zone_steps = 0
        conflict_steps = 0
        high_speed_zone_steps = 0
        creep_speed_steps = 0
        zone_speed_sum = 0.0
        action_change_sum = 0.0
        previous_action = 1
        ego_speed_sum = 0.0
        ego_stop_steps = 0
        ego_delay_sum = 0.0
        traffic_vehicle_count_sum = 0
        traffic_speed_sum = 0.0
        traffic_speed_samples = 0
        traffic_stop_vehicle_steps = 0
        traffic_delay_sum = 0.0
        action_counts = {0: 0, 1: 0, 2: 0}
        max_steps = int(math.ceil(float(cfg.duration) * float(cfg.policy_frequency))) + 5

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(np.asarray(action).reshape(-1)[0])
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)

            state = ego_route_state(env)
            speed = float(state["speed"])
            dist_center = float(state["dist_center"])
            current_ttc = route_min_ttc(env)
            current_conflict_ttc = route_min_conflict_ttc(env)
            if math.isfinite(current_ttc):
                min_ttc_seen = min(min_ttc_seen, current_ttc)
            if math.isfinite(current_conflict_ttc):
                min_conflict_ttc_seen = min(min_conflict_ttc_seen, current_conflict_ttc)

            in_creep_zone = dist_center <= 22.0 and not has_arrived(env, info)
            in_conflict_zone = dist_center <= 15.0 and not has_arrived(env, info)
            if in_creep_zone:
                zone_steps += 1
                zone_speed_sum += speed
                high_speed_zone_steps += int(speed > 5.0)
                creep_speed_steps += int(0.8 <= speed <= 4.8)
            if in_conflict_zone:
                conflict_steps += 1

            action_change_sum += float(action_int != previous_action)
            previous_action = action_int
            ego_speed_sum += speed
            ego_stop_steps += int(speed < 0.5)
            ego_delay_sum += max(8.0 - speed, 0.0)

            traffic_speeds = []
            for vehicle in getattr(env.unwrapped.road, "vehicles", []):
                if vehicle is getattr(env.unwrapped, "vehicle", None):
                    continue
                vehicle_speed = float(getattr(vehicle, "speed", 0.0))
                traffic_speeds.append(vehicle_speed)
            traffic_vehicle_count_sum += len(traffic_speeds)
            for vehicle_speed in traffic_speeds:
                traffic_speed_sum += vehicle_speed
                traffic_speed_samples += 1
                traffic_stop_vehicle_steps += int(vehicle_speed < 0.5)
                traffic_delay_sum += max(8.0 - vehicle_speed, 0.0)

            collided = collided or bool(info.get("crashed", False))
            arrived = arrived or bool(has_arrived(env, info))

        rows.append(
            {
                "maneuver": cfg.maneuver,
                "agent": agent_name,
                "episode": episode,
                "return": total_reward,
                "steps": steps,
                "survival_time_s": steps / cfg.policy_frequency,
                "collided": collided,
                "collision_count": int(collided),
                "success": bool(arrived and not collided),
                "timeout": bool(not collided and not arrived),
                "time_to_collision_s": steps / cfg.policy_frequency if collided else np.nan,
                "min_ttc": min_ttc_seen if math.isfinite(min_ttc_seen) else np.nan,
                "min_conflict_ttc": min_conflict_ttc_seen if math.isfinite(min_conflict_ttc_seen) else np.nan,
                "creep_zone_mean_speed": zone_speed_sum / max(zone_steps, 1),
                "high_speed_zone_rate": high_speed_zone_steps / max(zone_steps, 1),
                "creep_speed_rate": creep_speed_steps / max(zone_steps, 1),
                "conflict_steps": conflict_steps,
                "mean_action_change_rate": action_change_sum / max(steps, 1),
                "ego_mean_speed": ego_speed_sum / max(steps, 1),
                "ego_stop_rate": ego_stop_steps / max(steps, 1),
                "ego_stopped_time_s": ego_stop_steps / cfg.policy_frequency,
                "ego_delay_proxy_s": ego_delay_sum / cfg.policy_frequency,
                "traffic_vehicle_seconds": traffic_vehicle_count_sum / cfg.policy_frequency,
                "traffic_mean_speed": traffic_speed_sum / max(traffic_speed_samples, 1),
                "traffic_stop_rate": traffic_stop_vehicle_steps / max(traffic_speed_samples, 1),
                "traffic_delay_proxy_s": traffic_delay_sum / cfg.policy_frequency,
                "system_delay_proxy_s": (ego_delay_sum + traffic_delay_sum) / cfg.policy_frequency,
                "slower_action_rate": action_counts.get(0, 0) / max(steps, 1),
                "idle_action_rate": action_counts.get(1, 0) / max(steps, 1),
                "faster_action_rate": action_counts.get(2, 0) / max(steps, 1),
                "safety_shield": False,
                "safety_intervention_rate": 0.0,
                "safety_emergency_brake_rate": 0.0,
                "safety_progress_assist_rate": 0.0,
                "safety_violation_rate": 0.0,
                "safety_min_h": np.nan,
                "safety_neighbor_rate": 0.0,
            }
        )
        env.close()

    df = pd.DataFrame(rows)
    return df, summarize_route_metrics(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train native HighwayEnv intersection PPO baselines.")
    parser.add_argument("--results-dir", type=Path, default=Path("notebooks/results/nativeIntersectionPPO12_05"))
    parser.add_argument("--maneuvers", nargs="+", default=["straight", "left"], choices=["straight", "left"])
    parser.add_argument("--initial-vehicles", type=int, default=12)
    parser.add_argument("--spawn-probability", type=float, default=0.5)
    parser.add_argument("--duration", type=int, default=22)
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--road-class", choices=["native", "safe_regulated", "plain"], default="plain")
    parser.add_argument("--safe-lane-lookup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_intersection_road_class(args.road_class)
    if args.safe_lane_lookup:
        patch_closest_lane_lookup()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "models").mkdir(exist_ok=True)
    config = vars(args).copy()
    config["results_dir"] = str(args.results_dir)
    config["road_class"] = args.road_class
    config["safe_lane_lookup"] = args.safe_lane_lookup
    (args.results_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_eval = []
    all_summary = []
    for maneuver_index, maneuver in enumerate(args.maneuvers):
        cfg = config_for(maneuver, args)
        model_path = args.results_dir / "models" / f"{maneuver}_native_intersection_ppo_{args.timesteps}.zip"
        start = time.time()
        if args.skip_train:
            from stable_baselines3 import PPO

            print(f"\n=== {maneuver.upper()} load native PPO ===", flush=True)
            model = PPO.load(model_path, device=args.device)
        elif args.resume_training:
            print(f"\n=== {maneuver.upper()} resume/train native PPO chunk ===", flush=True)
            print(json.dumps(asdict(cfg), indent=2), flush=True)
            model = resume_train_ppo(
                cfg=cfg,
                model_path=model_path,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                seed=args.seed + 1000 * maneuver_index,
                device=args.device,
            )
        elif model_path.exists() and not args.overwrite:
            from stable_baselines3 import PPO

            print(f"\n=== {maneuver.upper()} load native PPO ===", flush=True)
            model = PPO.load(model_path, device=args.device)
        else:
            print(f"\n=== {maneuver.upper()} train native PPO ===", flush=True)
            print(json.dumps(asdict(cfg), indent=2), flush=True)
            model = train_ppo(
                cfg=cfg,
                model_path=model_path,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                seed=args.seed + 1000 * maneuver_index,
                device=args.device,
            )
        train_wall_time_s = time.time() - start
        if args.skip_eval:
            continue
        print(f"\n=== {maneuver.upper()} evaluate native PPO ===", flush=True)
        eval_df, summary = evaluate_ppo(
            model,
            cfg=cfg,
            episodes=args.eval_episodes,
            seed=args.seed + 5000 + 1000 * maneuver_index,
            agent_name="native_intersection_ppo",
        )
        eval_df["model_path"] = str(model_path)
        eval_df["initial_vehicle_count"] = cfg.initial_vehicle_count
        eval_df["spawn_probability"] = cfg.spawn_probability
        eval_df["duration"] = cfg.duration
        eval_df["reward_formulation"] = "native_highway_env_intersection"
        summary["model_path"] = str(model_path)
        summary["train_wall_time_s"] = train_wall_time_s
        summary["initial_vehicle_count"] = cfg.initial_vehicle_count
        summary["spawn_probability"] = cfg.spawn_probability
        summary["duration"] = cfg.duration
        summary["reward_formulation"] = "native_highway_env_intersection"
        eval_df.to_csv(args.results_dir / f"{maneuver}_native_intersection_ppo_eval_episodes.csv", index=False)
        summary.to_csv(args.results_dir / f"{maneuver}_native_intersection_ppo_summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)
        all_eval.append(eval_df)
        all_summary.append(summary)

    if not all_eval or not all_summary:
        return
    combined_eval = pd.concat(all_eval, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_eval.to_csv(args.results_dir / "combined_eval_episodes.csv", index=False)
    combined_summary.to_csv(args.results_dir / "combined_summary.csv", index=False)
    print("\n=== COMBINED SUMMARY ===", flush=True)
    print(combined_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
