from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

Variant = Literal["standard", "creeping", "creeping_residual", "creeping_negotiation"]


@dataclass(frozen=True)
class HighwayIntersectionConfig:
    policy_frequency: int = 5
    simulation_frequency: int = 15
    duration: int = 20
    initial_vehicle_count: int = 6
    spawn_probability: float = 0.30
    destination: str = "o1"
    max_speed: float = 12.0
    accel_min: float = -4.0
    accel_max: float = 2.0
    approach_radius: float = 38.0
    creep_radius: float = 18.0
    high_speed_zone_threshold: float = 5.0
    creep_speed_low: float = 1.0
    creep_speed_high: float = 4.5
    ttc_critical: float = 2.0
    ttc_caution: float = 5.0


DEFAULT_CONFIG = HighwayIntersectionConfig()

RAW_OBS_SHAPE = (15, 7)
FEATURE_OBS_SIZE = 15 * 7 + 6


def highway_env_config(cfg: HighwayIntersectionConfig = DEFAULT_CONFIG) -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "normalize": False,
            "flatten": False,
            "observe_intentions": False,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": False,
            "acceleration_range": [cfg.accel_min, cfg.accel_max],
            "speed_range": [0.0, cfg.max_speed],
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
        "reward_speed_range": [7.0, 9.0],
        "normalize_reward": False,
        "offroad_terminal": False,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 7.15,
        "render_agent": True,
        "offscreen_rendering": False,
    }


def _vehicle_velocity(vehicle) -> np.ndarray:
    velocity = getattr(vehicle, "velocity", None)
    if velocity is not None:
        return np.asarray(velocity, dtype=np.float32)
    speed = float(getattr(vehicle, "speed", 0.0))
    heading = float(getattr(vehicle, "heading", 0.0))
    return np.asarray([speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float32)


def min_time_to_collision(env, max_distance: float = 60.0) -> float:
    ego = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if ego is None or road is None:
        return math.inf

    ego_position = np.asarray(ego.position, dtype=np.float32)
    ego_velocity = _vehicle_velocity(ego)
    best_ttc = math.inf

    for other in road.vehicles:
        if other is ego:
            continue
        rel_position = np.asarray(other.position, dtype=np.float32) - ego_position
        distance = float(np.linalg.norm(rel_position))
        if distance < 1e-3 or distance > max_distance:
            continue
        rel_velocity = _vehicle_velocity(other) - ego_velocity
        closing_speed = -float(np.dot(rel_position, rel_velocity)) / distance
        if closing_speed <= 1e-3:
            continue
        best_ttc = min(best_ttc, distance / closing_speed)

    return best_ttc


def has_arrived(env, info: dict) -> bool:
    rewards = info.get("rewards", {}) if isinstance(info, dict) else {}
    if bool(rewards.get("arrived_reward", False)):
        return True
    vehicle = getattr(env.unwrapped, "vehicle", None)
    has_arrived_fn = getattr(env.unwrapped, "has_arrived", None)
    if vehicle is not None and callable(has_arrived_fn):
        try:
            return bool(has_arrived_fn(vehicle))
        except Exception:
            return False
    return False


def ego_state(env) -> tuple[float, float]:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return 0.0, math.inf
    speed = float(getattr(vehicle, "speed", 0.0))
    dist_center = float(np.linalg.norm(np.asarray(vehicle.position, dtype=np.float32)))
    return speed, dist_center


def ego_position_xy(env) -> tuple[float, float]:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return 0.0, math.inf
    position = np.asarray(vehicle.position, dtype=np.float32)
    return float(position[0]), float(position[1])


def negotiation_target_speed(y: float, ttc: float) -> float:
    """Reward-only target speed profile used to score behavior, not to override actions."""
    if y < -6.0:
        return 8.5
    if y < 16.0:
        if math.isfinite(ttc) and ttc < 1.8:
            return 1.2
        if math.isfinite(ttc) and ttc < 3.0:
            return 2.4
        return 3.8
    if y < 34.0:
        if math.isfinite(ttc) and ttc < 2.5:
            return 3.0
        return 6.2
    return 8.5


def normalize_kinematics_observation(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(RAW_OBS_SHAPE).copy()
    arr[:, 1] /= 100.0
    arr[:, 2] /= 100.0
    arr[:, 3] /= 20.0
    arr[:, 4] /= 20.0
    return arr


def decode_ego_from_observation(obs: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1 and arr.size >= 15 * 7:
        ego = arr[: 15 * 7].reshape(RAW_OBS_SHAPE)[0].copy()
        x = float(ego[1] * 100.0)
        y = float(ego[2] * 100.0)
        vx = float(ego[3] * 20.0)
        vy = float(ego[4] * 20.0)
    else:
        ego = arr.reshape(RAW_OBS_SHAPE)[0]
        x = float(ego[1])
        y = float(ego[2])
        vx = float(ego[3])
        vy = float(ego[4])
    return x, y, vx, vy


class NegotiationObservationWrapper(gym.ObservationWrapper):
    """Flatten kinematics and append reward-only negotiation cues."""

    def __init__(self, env: gym.Env, cfg: HighwayIntersectionConfig = DEFAULT_CONFIG):
        super().__init__(env)
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(FEATURE_OBS_SIZE,),
            dtype=np.float32,
        )

    def observation(self, observation):
        normalized = normalize_kinematics_observation(observation).reshape(-1)
        speed, _ = ego_state(self.env)
        _, y = ego_position_xy(self.env)
        ttc = min_time_to_collision(self.env)
        ttc_feature = min(ttc, 10.0) / 10.0 if math.isfinite(ttc) else 1.0
        target_speed = negotiation_target_speed(y, ttc)
        aux = np.asarray(
            [
                np.clip(speed / self.cfg.max_speed, 0.0, 1.5),
                np.clip(y / 50.0, -1.5, 1.5),
                ttc_feature,
                float(y <= self.cfg.approach_radius and y >= -6.0),
                float(y <= self.cfg.creep_radius and y >= -6.0),
                np.clip(target_speed / self.cfg.max_speed, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([normalized, aux], dtype=np.float32)


class HighwayCreepingRewardWrapper(gym.Wrapper):
    """HighwayEnv intersection wrapper for comparing fast vs creeping throttle behavior."""

    def __init__(self, env: gym.Env, variant: Variant, cfg: HighwayIntersectionConfig = DEFAULT_CONFIG):
        super().__init__(env)
        self.variant = variant
        self.cfg = cfg
        self.previous_distance = math.inf
        self.previous_y = math.inf
        self.previous_action = 0.0
        self.zone_steps = 0
        self.high_speed_zone_steps = 0
        self.creep_speed_steps = 0
        self.zone_speed_sum = 0.0
        self.abs_action_sum = 0.0
        self.abs_delta_action_sum = 0.0
        self.min_ttc_seen = math.inf

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        _, dist_center = ego_state(self.env)
        _, y = ego_position_xy(self.env)
        self.previous_distance = dist_center
        self.previous_y = y
        self.previous_action = 0.0
        self.zone_steps = 0
        self.high_speed_zone_steps = 0
        self.creep_speed_steps = 0
        self.zone_speed_sum = 0.0
        self.abs_action_sum = 0.0
        self.abs_delta_action_sum = 0.0
        self.min_ttc_seen = math.inf
        return obs, info

    def step(self, action):
        action_scalar = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})

        speed, dist_center = ego_state(self.env)
        _, y = ego_position_xy(self.env)
        ttc = min_time_to_collision(self.env)
        if math.isfinite(ttc):
            self.min_ttc_seen = min(self.min_ttc_seen, ttc)

        has_cleared_conflict = y < -6.0
        in_approach = (not has_cleared_conflict) and y <= self.cfg.approach_radius
        in_creep_zone = (not has_cleared_conflict) and y <= self.cfg.creep_radius
        arrived = has_arrived(self.env, info)
        crashed = bool(info.get("crashed", False))

        self.abs_action_sum += abs(action_scalar)
        self.abs_delta_action_sum += abs(action_scalar - self.previous_action)
        self.previous_action = action_scalar

        if in_creep_zone:
            self.zone_steps += 1
            self.zone_speed_sum += speed
            self.high_speed_zone_steps += int(speed > self.cfg.high_speed_zone_threshold)
            self.creep_speed_steps += int(self.cfg.creep_speed_low <= speed <= self.cfg.creep_speed_high)

        if self.variant == "standard":
            reward = self._standard_reward(speed, arrived, crashed)
        elif self.variant in {"creeping", "creeping_negotiation"}:
            reward = self._creeping_reward(speed, dist_center, ttc, arrived, crashed, action_scalar)
        else:
            reward = float(base_reward)

        if truncated and not arrived and not crashed:
            reward -= 25.0 if self.variant in {"creeping", "creeping_negotiation"} else 14.0

        step_count = max(1, int(round(self.env.unwrapped.time * self.cfg.policy_frequency)))
        info.update(
            {
                "arrived": arrived,
                "ttc": ttc,
                "dist_center": dist_center,
                "ego_y": y,
                "in_creep_zone": in_creep_zone,
                "ego_speed": speed,
                "throttle": action_scalar,
                "zone_steps": self.zone_steps,
                "creep_zone_mean_speed": self.zone_speed_sum / max(self.zone_steps, 1),
                "high_speed_zone_rate": self.high_speed_zone_steps / max(self.zone_steps, 1),
                "creep_speed_rate": self.creep_speed_steps / max(self.zone_steps, 1),
                "mean_abs_throttle": self.abs_action_sum / step_count,
                "mean_abs_delta_throttle": self.abs_delta_action_sum / step_count,
                "min_ttc_seen": self.min_ttc_seen,
                "reward_target_speed": negotiation_target_speed(y, ttc),
                "shaped_reward": reward,
            }
        )
        self.previous_distance = dist_center
        self.previous_y = y
        return obs, float(reward), terminated, truncated, info

    def _standard_reward(self, speed: float, arrived: bool, crashed: bool) -> float:
        reward = -0.02 + 0.10 * speed
        if arrived:
            reward += 32.0
        if crashed:
            reward -= 25.0
        return reward

    def _creeping_reward(self, speed: float, dist_center: float, ttc: float, arrived: bool, crashed: bool, action: float) -> float:
        cfg = self.cfg
        _, y = ego_position_xy(self.env)
        has_cleared_conflict = y < -6.0
        in_approach = (not has_cleared_conflict) and y <= cfg.approach_radius
        in_creep_zone = (not has_cleared_conflict) and y <= cfg.creep_radius
        progress = max(0.0, self.previous_y - y)
        target_speed = negotiation_target_speed(y, ttc)
        speed_error = abs(speed - target_speed)
        speed_match = math.exp(-0.5 * (speed_error / 1.7) ** 2)

        reward = -0.03
        if in_approach:
            reward += 0.95 * progress * (0.35 + 0.65 * speed_match)
            if speed < 0.6 and not (math.isfinite(ttc) and ttc < 1.4):
                reward -= 0.45
        else:
            reward += 0.16 * speed

        if in_creep_zone:
            reward += 0.90 * speed_match
            if cfg.creep_speed_low <= speed <= cfg.creep_speed_high:
                reward += 0.35
            if speed > cfg.high_speed_zone_threshold:
                reward -= 0.28 * (speed - cfg.high_speed_zone_threshold)

        if math.isfinite(ttc):
            if ttc < cfg.ttc_critical:
                reward -= 3.5 * (cfg.ttc_critical - ttc) / cfg.ttc_critical
                reward -= 0.18 * max(speed - 1.0, 0.0)
            elif in_creep_zone and ttc < cfg.ttc_caution:
                reward += 0.25 if speed <= cfg.creep_speed_high else -0.30

        if in_creep_zone and y < 4.0 and (not math.isfinite(ttc) or ttc > 2.2):
            reward += 0.70 * progress
            reward += 0.06 * speed

        reward -= 0.025 * abs(action - self.previous_action)
        if arrived:
            reward += 55.0
        if crashed:
            reward -= 45.0
        return reward


def creeping_prior_throttle(env, cfg: HighwayIntersectionConfig = DEFAULT_CONFIG) -> tuple[float, float]:
    speed, _ = ego_state(env)
    _, y = ego_position_xy(env)
    if y < -6.0:
        target_speed = 8.5
    elif y < 16.0:
        target_speed = 3.5
    elif y < 34.0:
        target_speed = 6.5
    else:
        target_speed = 8.5
    throttle = float(np.clip((target_speed - speed) / 3.0, -1.0, 1.0))
    return throttle, target_speed


class ResidualCreepingActionWrapper(gym.ActionWrapper):
    """Let PPO learn a residual throttle around a creeping target-speed prior."""

    def __init__(self, env: gym.Env, cfg: HighwayIntersectionConfig = DEFAULT_CONFIG, residual_scale: float = 0.35):
        super().__init__(env)
        self.cfg = cfg
        self.residual_scale = float(residual_scale)

    def step(self, action):
        residual = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        prior, target_speed = creeping_prior_throttle(self.env, self.cfg)
        applied = float(np.clip(prior + self.residual_scale * residual, -1.0, 1.0))
        obs, reward, terminated, truncated, info = self.env.step(np.asarray([applied], dtype=np.float32))
        info = dict(info or {})
        info.update(
            {
                "residual_throttle": residual,
                "prior_throttle": prior,
                "applied_throttle": applied,
                "target_speed": target_speed,
            }
        )
        return obs, reward, terminated, truncated, info


def make_intersection_env(
    variant: Variant = "creeping",
    seed: int | None = None,
    render_mode: str | None = None,
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    use_features: bool = False,
) -> gym.Env:
    env = gym.make("intersection-v0", render_mode=render_mode)
    env.unwrapped.configure(highway_env_config(cfg))
    reward_variant: Variant = "creeping" if variant == "creeping_residual" else variant
    env = HighwayCreepingRewardWrapper(env, variant=reward_variant, cfg=cfg)
    if variant == "creeping_residual":
        env = ResidualCreepingActionWrapper(env, cfg=cfg)
    if use_features:
        env = NegotiationObservationWrapper(env, cfg=cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(
    variant: Variant,
    n_envs: int,
    seed: int,
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    backend: Literal["dummy", "subproc"] = "dummy",
    use_features: bool = False,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    def thunk(rank: int):
        def _init():
            return make_intersection_env(variant=variant, seed=seed + rank, cfg=cfg, use_features=use_features)

        return _init

    env_fns = [thunk(rank) for rank in range(n_envs)]
    vec_cls = SubprocVecEnv if backend == "subproc" else DummyVecEnv
    return VecMonitor(vec_cls(env_fns))


def train_ppo(
    variant: Variant,
    total_timesteps: int,
    model_path: Path,
    n_envs: int = 8,
    seed: int = 7,
    device: str = "auto",
    vec_backend: Literal["dummy", "subproc"] = "dummy",
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    use_features: bool = False,
):
    from stable_baselines3 import PPO

    env = make_vec_env(variant=variant, n_envs=n_envs, seed=seed, cfg=cfg, backend=vec_backend, use_features=use_features)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=min(1024, 256 * n_envs),
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.92,
        learning_rate=3e-4,
        ent_coef=0.01,
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


def finetune_ppo(
    base_model_path: Path,
    variant: Variant,
    total_timesteps: int,
    model_path: Path,
    n_envs: int = 8,
    seed: int = 7,
    device: str = "auto",
    vec_backend: Literal["dummy", "subproc"] = "dummy",
    learning_rate: float | None = None,
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    use_features: bool = False,
):
    from stable_baselines3 import PPO

    env = make_vec_env(variant=variant, n_envs=n_envs, seed=seed, cfg=cfg, backend=vec_backend, use_features=use_features)
    model = PPO.load(str(base_model_path), env=env, device=device)
    if learning_rate is not None:
        model.learning_rate = learning_rate
        model.lr_schedule = lambda _: learning_rate
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def summarize_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for agent, group in df.groupby("agent", dropna=False):
        collided = group["collided"].astype(bool)
        collision_times = group.loc[collided, "time_to_collision_s"].dropna()
        rows.append(
            {
                "agent": agent,
                "episodes": len(group),
                "success_rate": group["success"].mean(),
                "collision_rate": collided.mean(),
                "average_collisions": group["collision_count"].mean(),
                "mean_time_to_collision_s": collision_times.mean() if len(collision_times) else np.nan,
                "median_time_to_collision_s": collision_times.median() if len(collision_times) else np.nan,
                "mean_survival_time_s": group["survival_time_s"].mean(),
                "mean_return": group["return"].mean(),
                "mean_creep_zone_speed": group["creep_zone_mean_speed"].mean(),
                "high_speed_zone_rate": group["high_speed_zone_rate"].mean(),
                "creep_speed_rate": group["creep_speed_rate"].mean(),
                "mean_abs_throttle": group["mean_abs_throttle"].mean(),
                "mean_abs_delta_throttle": group["mean_abs_delta_throttle"].mean(),
                "mean_min_ttc": group["min_ttc"].replace(np.inf, np.nan).mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_agent(
    model,
    agent_name: str,
    variant: Variant,
    episodes: int = 50,
    seed: int = 1000,
    deterministic: bool = True,
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    use_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for episode in range(episodes):
        env = make_intersection_env(variant=variant, seed=seed + episode, cfg=cfg, use_features=use_features)
        obs, info = env.reset(seed=seed + episode)
        done = False
        steps = 0
        total_reward = 0.0
        collided = False
        arrived = False
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)
            collided = collided or bool(info.get("crashed", False))
            arrived = arrived or bool(info.get("arrived", False))
            final_info = info

        rows.append(
            {
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
                "min_ttc": final_info.get("min_ttc_seen", np.nan),
                "creep_zone_mean_speed": final_info.get("creep_zone_mean_speed", np.nan),
                "high_speed_zone_rate": final_info.get("high_speed_zone_rate", np.nan),
                "creep_speed_rate": final_info.get("creep_speed_rate", np.nan),
                "mean_abs_throttle": final_info.get("mean_abs_throttle", np.nan),
                "mean_abs_delta_throttle": final_info.get("mean_abs_delta_throttle", np.nan),
            }
        )
        env.close()

    df = pd.DataFrame(rows)
    return df, summarize_episode_metrics(df)


class ConstantThrottlePolicy:
    def __init__(self, throttle: float):
        self.throttle = float(throttle)

    def predict(self, obs, deterministic: bool = True):
        return np.asarray([self.throttle], dtype=np.float32), None


class TrafficAwareCreepingPolicy:
    """Reference policy used only for curriculum distillation, not as a runtime constraint."""

    def predict(self, obs, deterministic: bool = True):
        x, y, vx, vy = decode_ego_from_observation(obs)
        speed = math.hypot(vx, vy)
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= FEATURE_OBS_SIZE:
            ttc_feature = float(arr[-4])
            ttc = math.inf if ttc_feature >= 0.999 else ttc_feature * 10.0
        else:
            ttc = math.inf

        if y < 4.0:
            target_speed = 8.5
        elif y < 16.0:
            target_speed = 1.5 if ttc < 2.3 else 3.8
        elif y < 34.0:
            target_speed = 3.5 if ttc < 2.3 else 6.5
        else:
            target_speed = 8.5

        throttle = float(np.clip((target_speed - speed) / 3.0, -1.0, 1.0))
        return np.asarray([throttle], dtype=np.float32), None


class ScriptedCreepingPolicy:
    def predict(self, obs, deterministic: bool = True):
        x, y, vx, vy = decode_ego_from_observation(obs)
        speed = math.hypot(vx, vy)
        dist_center = math.hypot(x, y)
        if y < -6.0:
            target_speed = 8.5
        elif y < 16.0:
            target_speed = 3.0
        elif y < 34.0:
            target_speed = 5.5
        else:
            target_speed = 8.0
        error = target_speed - speed
        throttle = float(np.clip(error / 4.0, -1.0, 1.0))
        return np.asarray([throttle], dtype=np.float32), None


def collect_teacher_dataset(
    teacher,
    variant: Variant = "creeping_negotiation",
    episodes: int = 100,
    seed: int = 26000,
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
    use_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []
    for episode in range(episodes):
        env = make_intersection_env(variant=variant, seed=seed + episode, cfg=cfg, use_features=use_features)
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            action, _ = teacher.predict(obs, deterministic=True)
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def behavior_clone_ppo_actor(
    base_model_path: Path,
    teacher,
    model_path: Path,
    episodes: int = 100,
    seed: int = 26000,
    epochs: int = 80,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    device: str = "auto",
    cfg: HighwayIntersectionConfig = DEFAULT_CONFIG,
):
    import torch
    from stable_baselines3 import PPO

    obs_np, act_np = collect_teacher_dataset(
        teacher,
        variant="creeping_negotiation",
        episodes=episodes,
        seed=seed,
        cfg=cfg,
        use_features=True,
    )

    model = PPO.load(str(base_model_path), device=device)
    policy = model.policy
    policy.train()
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(act_np, dtype=torch.float32, device=model.device)
    n_samples = obs_t.shape[0]

    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(n_samples, device=model.device)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            idx = permutation[start : start + batch_size]
            features = policy.extract_features(obs_t[idx])
            latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)
            loss = torch.nn.functional.mse_loss(mean_actions, act_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    history = pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "bc_loss": losses})
    return model, history


def evaluate_reference_policies(
    episodes: int = 50,
    seed: int = 3000,
    use_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs: list[tuple[str, object, Variant]] = [
        ("scripted_non_creeping_fast", ConstantThrottlePolicy(0.35), "standard"),
        ("scripted_creeping", ScriptedCreepingPolicy(), "creeping"),
        ("traffic_aware_creeping_teacher", TrafficAwareCreepingPolicy(), "creeping_negotiation"),
        ("zero_residual_creeping_prior", ConstantThrottlePolicy(0.0), "creeping_residual"),
    ]
    frames = []
    summaries = []
    for name, policy, variant in specs:
        df, summary = evaluate_agent(policy, name, variant=variant, episodes=episodes, seed=seed, use_features=use_features)
        frames.append(df)
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.concat(summaries, ignore_index=True)
