from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


RouteVariant = Literal["standard", "creeping"]
Maneuver = Literal["straight", "left"]
VecBackend = Literal["dummy", "subproc"]

ROUTE_VEHICLE_COUNT = 15
ROUTE_RAW_FEATURES = 7
ATTENTION_TOKEN_FEATURES = 9
ATTENTION_AUX_FEATURES = 6
ATTENTION_OBS_SIZE = ROUTE_VEHICLE_COUNT * ATTENTION_TOKEN_FEATURES + ATTENTION_AUX_FEATURES


@dataclass(frozen=True)
class RouteIntersectionConfig:
    maneuver: Maneuver = "straight"
    destination: str = "o2"
    policy_frequency: int = 5
    simulation_frequency: int = 15
    duration: int = 22
    initial_vehicle_count: int = 6
    spawn_probability: float = 0.35
    vehicles_count: int = ROUTE_VEHICLE_COUNT
    target_speeds: tuple[float, ...] = (0.0, 2.0, 4.5, 7.0, 9.0)
    max_speed: float = 9.0
    approach_radius: float = 36.0
    creep_radius: float = 22.0
    conflict_radius: float = 15.0
    high_speed_zone_threshold: float = 5.0
    creep_speed_low: float = 0.8
    creep_speed_high: float = 4.8
    ttc_critical: float = 2.0
    ttc_caution: float = 4.5
    negotiating_traffic: bool = False
    yield_ego_speed_threshold: float = 5.0
    yield_radius: float = 42.0
    yield_target_speed: float = 2.0
    yield_time_window: float = 6.0
    yield_clear_conflict: bool = False
    safety_shield: bool = False
    safety_ellipse_margin: float = 1.2
    safety_min_clearance: float = 0.15
    safety_neighbor_range: float = 70.0
    safety_apply_radius: float = 45.0
    safety_conflict_radius: float = 24.0
    safety_near_field_radius: float = 18.0
    safety_near_field_horizon: float = 1.0
    safety_time_gate: bool = False
    safety_time_gate_buffer: float = 0.4
    safety_prediction_horizon: float = 4.8
    safety_prediction_dt: float = 0.2
    safety_max_accel: float = 4.0
    safety_action_preference: tuple[int, ...] = (1, 0, 2)
    safety_emergency_brake: bool = True
    safety_emergency_decel: float = 7.0
    safety_progress_assist: bool = False
    safety_progress_assist_radius: float = 55.0
    interaction_reward_focus: bool = False
    efficiency_reference_speed: float = 8.0
    efficiency_stop_speed_threshold: float = 0.5


def route_config_for(maneuver: Maneuver, **overrides) -> RouteIntersectionConfig:
    destination = "o2" if maneuver == "straight" else "o1"
    return replace(RouteIntersectionConfig(maneuver=maneuver, destination=destination), **overrides)


def highway_route_env_config(cfg: RouteIntersectionConfig) -> dict:
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
            "normalize": False,
            "flatten": False,
            "observe_intentions": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": list(cfg.target_speeds),
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


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def inflated_ellipse_axes(length: float, width: float, margin: float) -> tuple[float, float]:
    """Inflated ellipse semi-axes covering a vehicle footprint."""
    a = float(length) / math.sqrt(2.0) + 2.0 * float(margin)
    b = float(width) / math.sqrt(2.0) + 2.0 * float(margin)
    return max(a, 1e-6), max(b, 1e-6)


def ellipse_radius_along_line(a: float, b: float, delta: float) -> float:
    cos_delta = math.cos(float(delta))
    sin_delta = math.sin(float(delta))
    denom = math.sqrt((float(b) * cos_delta) ** 2 + (float(a) * sin_delta) ** 2)
    return float(float(a) * float(b) / max(denom, 1e-9))


def vehicle_ellipse_clearance(
    ego_position: np.ndarray,
    ego_heading: float,
    ego_length: float,
    ego_width: float,
    other_position: np.ndarray,
    other_heading: float,
    other_length: float,
    other_width: float,
    margin: float,
) -> tuple[float, float, float]:
    """Centerline ellipse clearance h = center distance - required distance."""
    delta = np.asarray(other_position, dtype=np.float32) - np.asarray(ego_position, dtype=np.float32)
    center_distance = float(np.linalg.norm(delta))
    phi = 0.0 if center_distance < 1e-9 else math.atan2(float(delta[1]), float(delta[0]))
    ego_a, ego_b = inflated_ellipse_axes(ego_length, ego_width, margin)
    other_a, other_b = inflated_ellipse_axes(other_length, other_width, margin)
    ego_radius = ellipse_radius_along_line(ego_a, ego_b, _wrap_angle(phi - ego_heading))
    other_radius = ellipse_radius_along_line(other_a, other_b, _wrap_angle(phi - other_heading))
    required_distance = ego_radius + other_radius
    return center_distance - required_distance, center_distance, required_distance


def route_min_ttc(env: gym.Env, max_distance: float = 70.0) -> float:
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


def _time_to_intersection_center(position: np.ndarray, velocity: np.ndarray) -> float:
    dist_center = float(np.linalg.norm(position))
    if dist_center < 1e-3:
        return 0.0
    approach_speed = -float(np.dot(position, velocity)) / dist_center
    if approach_speed <= 1e-3:
        return math.inf
    return dist_center / approach_speed


def route_min_conflict_ttc(
    env: gym.Env,
    max_distance: float = 70.0,
    overlap_window: float = 4.0,
    center_radius: float = 20.0,
) -> float:
    """Approximate risk only for vehicles competing for the intersection center."""
    ego = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if ego is None or road is None:
        return math.inf

    ego_position = np.asarray(ego.position, dtype=np.float32)
    ego_velocity = _vehicle_velocity(ego)
    ego_time = _time_to_intersection_center(ego_position, ego_velocity)
    if not math.isfinite(ego_time):
        return math.inf

    best_ttc = math.inf
    for other in road.vehicles:
        if other is ego:
            continue
        other_position = np.asarray(other.position, dtype=np.float32)
        other_distance = float(np.linalg.norm(other_position))
        if other_distance > max_distance:
            continue
        other_time = _time_to_intersection_center(other_position, _vehicle_velocity(other))
        if not math.isfinite(other_time):
            continue
        same_conflict_window = abs(ego_time - other_time) <= overlap_window
        already_in_center = (
            float(np.linalg.norm(ego_position)) <= center_radius
            and other_distance <= center_radius
        )
        if same_conflict_window or already_in_center:
            best_ttc = min(best_ttc, max(ego_time, other_time))
    return best_ttc


def raw_observation_ttc(obs: np.ndarray, max_distance: float = 70.0) -> float:
    raw = np.asarray(obs, dtype=np.float32).reshape(ROUTE_VEHICLE_COUNT, ROUTE_RAW_FEATURES)
    ego = raw[0]
    if ego[0] < 0.5:
        return math.inf
    ego_pos = ego[1:3]
    ego_vel = ego[3:5]
    best_ttc = math.inf
    for other in raw[1:]:
        if other[0] < 0.5:
            continue
        rel_position = other[1:3] - ego_pos
        distance = float(np.linalg.norm(rel_position))
        if distance < 1e-3 or distance > max_distance:
            continue
        rel_velocity = other[3:5] - ego_vel
        closing_speed = -float(np.dot(rel_position, rel_velocity)) / distance
        if closing_speed <= 1e-3:
            continue
        best_ttc = min(best_ttc, distance / closing_speed)
    return best_ttc


def raw_observation_conflict_ttc(
    obs: np.ndarray,
    max_distance: float = 70.0,
    overlap_window: float = 4.0,
    center_radius: float = 20.0,
) -> float:
    raw = np.asarray(obs, dtype=np.float32).reshape(ROUTE_VEHICLE_COUNT, ROUTE_RAW_FEATURES)
    ego = raw[0]
    if ego[0] < 0.5:
        return math.inf
    ego_position = ego[1:3]
    ego_time = _time_to_intersection_center(ego_position, ego[3:5])
    if not math.isfinite(ego_time):
        return math.inf
    best_ttc = math.inf
    for other in raw[1:]:
        if other[0] < 0.5:
            continue
        other_position = other[1:3]
        other_distance = float(np.linalg.norm(other_position))
        if other_distance > max_distance:
            continue
        other_time = _time_to_intersection_center(other_position, other[3:5])
        if not math.isfinite(other_time):
            continue
        same_conflict_window = abs(ego_time - other_time) <= overlap_window
        already_in_center = float(np.linalg.norm(ego_position)) <= center_radius and other_distance <= center_radius
        if same_conflict_window or already_in_center:
            best_ttc = min(best_ttc, max(ego_time, other_time))
    return best_ttc


def has_arrived(env: gym.Env, info: dict | None = None) -> bool:
    rewards = (info or {}).get("rewards", {}) if isinstance(info, dict) else {}
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


def ego_route_state(env: gym.Env) -> dict:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return {
            "speed": 0.0,
            "position": np.zeros(2, dtype=np.float32),
            "dist_center": math.inf,
            "exit_lane": False,
            "lane_index": None,
        }
    position = np.asarray(vehicle.position, dtype=np.float32)
    lane_index = getattr(vehicle, "lane_index", None)
    exit_lane = bool(
        lane_index
        and isinstance(lane_index, tuple)
        and str(lane_index[0]).startswith("il")
        and str(lane_index[1]).startswith("o")
    )
    return {
        "speed": float(getattr(vehicle, "speed", 0.0)),
        "position": position,
        "dist_center": float(np.linalg.norm(position)),
        "exit_lane": exit_lane,
        "lane_index": lane_index,
    }


def apply_creeping_negotiation_yield(env: gym.Env, cfg: RouteIntersectionConfig, action_int: int | None = None) -> None:
    """Let cross traffic yield only when ego is genuinely creeping near the conflict zone."""
    if not cfg.negotiating_traffic:
        return
    ego = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if ego is None or road is None:
        return

    ego_position = np.asarray(ego.position, dtype=np.float32)
    ego_speed = float(getattr(ego, "speed", 0.0))
    ego_dist = float(np.linalg.norm(ego_position))
    ego_exit_lane = bool(
        getattr(ego, "lane_index", None)
        and str(ego.lane_index[0]).startswith("il")
        and str(ego.lane_index[1]).startswith("o")
    )
    is_negotiating_action = action_int == 0
    is_creeping_speed = ego_speed <= cfg.yield_ego_speed_threshold
    if ego_exit_lane or ego_dist > cfg.yield_radius or not (is_negotiating_action or is_creeping_speed):
        return

    ego_time = _time_to_intersection_center(ego_position, _vehicle_velocity(ego))
    for other in list(road.vehicles):
        if other is ego:
            continue
        other_position = np.asarray(other.position, dtype=np.float32)
        other_dist = float(np.linalg.norm(other_position))
        if other_dist > cfg.yield_radius:
            continue
        other_time = _time_to_intersection_center(other_position, _vehicle_velocity(other))
        if not math.isfinite(other_time):
            continue
        overlaps = is_negotiating_action or (not math.isfinite(ego_time)) or abs(ego_time - other_time) <= cfg.yield_time_window
        already_conflicting = ego_dist <= cfg.conflict_radius and other_dist <= cfg.conflict_radius
        if overlaps or already_conflicting:
            if cfg.yield_clear_conflict and is_negotiating_action and other_dist <= cfg.conflict_radius + 4.0:
                try:
                    road.vehicles.remove(other)
                except ValueError:
                    pass
                continue
            if other_dist <= cfg.conflict_radius and ego_dist > cfg.conflict_radius:
                clear_speed = max(cfg.creep_speed_high + 1.0, 5.5)
                if hasattr(other, "target_speed"):
                    other.target_speed = max(float(getattr(other, "target_speed", clear_speed)), clear_speed)
                if hasattr(other, "speed"):
                    other.speed = max(float(getattr(other, "speed", clear_speed)), clear_speed)
                continue
            if hasattr(other, "target_speed"):
                other.target_speed = min(float(getattr(other, "target_speed", cfg.yield_target_speed)), cfg.yield_target_speed)
            if hasattr(other, "speed"):
                other.speed = min(float(getattr(other, "speed", cfg.yield_target_speed)), cfg.yield_target_speed)


def route_target_speed(dist_center: float, ttc: float, exit_lane: bool, arrived: bool, cfg: RouteIntersectionConfig) -> float:
    """Reward-only speed profile. It scores negotiation; it never overrides the action."""
    if arrived or (exit_lane and dist_center > cfg.conflict_radius):
        return 8.5
    if dist_center > cfg.approach_radius:
        return 8.0
    if dist_center > cfg.creep_radius:
        if cfg.interaction_reward_focus:
            if math.isfinite(ttc) and ttc < cfg.ttc_caution:
                return 3.8
            return 6.0
        if math.isfinite(ttc) and ttc < cfg.ttc_caution:
            return 4.0
        return 6.8
    if cfg.interaction_reward_focus:
        if math.isfinite(ttc) and ttc < cfg.ttc_critical:
            return 1.8
        if math.isfinite(ttc) and ttc < cfg.ttc_caution:
            return 3.2
        return min(4.6, cfg.creep_speed_high)
    if math.isfinite(ttc) and ttc < cfg.ttc_critical:
        return 2.0
    if math.isfinite(ttc) and ttc < cfg.ttc_caution:
        return 3.6
    return 5.4


class RouteAttentionObservationWrapper(gym.ObservationWrapper):
    """Near-vehicle tokens plus route-level negotiation cues for the attention policy."""

    def __init__(self, env: gym.Env, cfg: RouteIntersectionConfig):
        super().__init__(env)
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ATTENTION_OBS_SIZE,),
            dtype=np.float32,
        )

    def observation(self, observation):
        raw = np.asarray(observation, dtype=np.float32).reshape(self.cfg.vehicles_count, ROUTE_RAW_FEATURES)
        ego = raw[0].copy()
        tokens = np.zeros((self.cfg.vehicles_count, ATTENTION_TOKEN_FEATURES), dtype=np.float32)
        if ego[0] < 0.5:
            return tokens.reshape(-1)

        ego_position = ego[1:3]
        ego_velocity = ego[3:5]
        tokens[0] = np.asarray(
            [
                1.0,
                0.0,
                0.0,
                np.clip(ego_velocity[0] / 20.0, -2.0, 2.0),
                np.clip(ego_velocity[1] / 20.0, -2.0, 2.0),
                ego[5],
                ego[6],
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        others = []
        for other in raw[1:]:
            if other[0] < 0.5:
                continue
            rel_position = other[1:3] - ego_position
            distance = float(np.linalg.norm(rel_position))
            rel_velocity = other[3:5] - ego_velocity
            closing_speed = 0.0
            ttc = math.inf
            if distance > 1e-3:
                closing_speed = max(0.0, -float(np.dot(rel_position, rel_velocity)) / distance)
                if closing_speed > 1e-3:
                    ttc = distance / closing_speed
            others.append((distance, rel_position, rel_velocity, other, ttc))

        others.sort(key=lambda item: item[0])
        for row, (distance, rel_position, rel_velocity, other, ttc) in enumerate(others[: self.cfg.vehicles_count - 1], start=1):
            ttc_feature = min(ttc, 10.0) / 10.0 if math.isfinite(ttc) else 1.0
            tokens[row] = np.asarray(
                [
                    1.0,
                    np.clip(rel_position[0] / 80.0, -2.0, 2.0),
                    np.clip(rel_position[1] / 80.0, -2.0, 2.0),
                    np.clip(rel_velocity[0] / 20.0, -2.0, 2.0),
                    np.clip(rel_velocity[1] / 20.0, -2.0, 2.0),
                    other[5],
                    other[6],
                    np.clip(distance / 80.0, 0.0, 2.0),
                    ttc_feature,
                ],
                dtype=np.float32,
            )

        state = ego_route_state(self.env)
        ttc = route_min_conflict_ttc(self.env)
        arrived = has_arrived(self.env, {})
        target_speed = route_target_speed(state["dist_center"], ttc, state["exit_lane"], arrived, self.cfg)
        ttc_feature = min(ttc, 10.0) / 10.0 if math.isfinite(ttc) else 1.0
        aux = np.asarray(
            [
                np.clip(state["speed"] / self.cfg.max_speed, 0.0, 1.5),
                np.clip(state["dist_center"] / 60.0, 0.0, 2.0),
                ttc_feature,
                float(state["dist_center"] <= self.cfg.approach_radius),
                float(state["dist_center"] <= self.cfg.creep_radius and not state["exit_lane"]),
                np.clip(target_speed / self.cfg.max_speed, 0.0, 1.2),
            ],
            dtype=np.float32,
        )
        return np.concatenate([tokens.reshape(-1), aux], dtype=np.float32)


class RouteCreepingRewardWrapper(gym.Wrapper):
    """Reward-only shaping for creeping/negotiation on routed straight and left-turn tasks."""

    def __init__(self, env: gym.Env, variant: RouteVariant, cfg: RouteIntersectionConfig):
        super().__init__(env)
        self.variant = variant
        self.cfg = cfg
        self.previous_dist_center = math.inf
        self.previous_action = 1
        self.zone_steps = 0
        self.conflict_steps = 0
        self.high_speed_zone_steps = 0
        self.creep_speed_steps = 0
        self.zone_speed_sum = 0.0
        self.action_change_sum = 0.0
        self.min_ttc_seen = math.inf
        self.min_conflict_ttc_seen = math.inf
        self.ego_speed_sum = 0.0
        self.ego_delay_sum = 0.0
        self.ego_stop_steps = 0
        self.traffic_vehicle_count_sum = 0
        self.traffic_speed_sum = 0.0
        self.traffic_speed_samples = 0
        self.traffic_stop_vehicle_steps = 0
        self.traffic_delay_sum = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = ego_route_state(self.env)
        self.previous_dist_center = state["dist_center"]
        self.previous_action = 1
        self.zone_steps = 0
        self.conflict_steps = 0
        self.high_speed_zone_steps = 0
        self.creep_speed_steps = 0
        self.zone_speed_sum = 0.0
        self.action_change_sum = 0.0
        self.min_ttc_seen = math.inf
        self.min_conflict_ttc_seen = math.inf
        self.ego_speed_sum = 0.0
        self.ego_delay_sum = 0.0
        self.ego_stop_steps = 0
        self.traffic_vehicle_count_sum = 0
        self.traffic_speed_sum = 0.0
        self.traffic_speed_samples = 0
        self.traffic_stop_vehicle_steps = 0
        self.traffic_delay_sum = 0.0
        return obs, info

    def step(self, action):
        action_int = int(np.asarray(action).reshape(-1)[0])
        apply_creeping_negotiation_yield(self.env, self.cfg, action_int=action_int)
        obs, base_reward, terminated, truncated, info = self.env.step(action_int)
        apply_creeping_negotiation_yield(self.env, self.cfg, action_int=action_int)
        info = dict(info or {})

        state = ego_route_state(self.env)
        speed = state["speed"]
        dist_center = state["dist_center"]
        exit_lane = state["exit_lane"]
        ttc = route_min_ttc(self.env)
        conflict_ttc = route_min_conflict_ttc(self.env)
        if math.isfinite(ttc):
            self.min_ttc_seen = min(self.min_ttc_seen, ttc)
        if math.isfinite(conflict_ttc):
            self.min_conflict_ttc_seen = min(self.min_conflict_ttc_seen, conflict_ttc)

        arrived = has_arrived(self.env, info)
        crashed = bool(info.get("crashed", False))
        in_creep_zone = dist_center <= self.cfg.creep_radius and not arrived
        in_conflict_zone = dist_center <= self.cfg.conflict_radius and not arrived
        target_speed = route_target_speed(dist_center, conflict_ttc, exit_lane, arrived, self.cfg)

        if in_creep_zone:
            self.zone_steps += 1
            self.zone_speed_sum += speed
            self.high_speed_zone_steps += int(speed > self.cfg.high_speed_zone_threshold)
            self.creep_speed_steps += int(self.cfg.creep_speed_low <= speed <= self.cfg.creep_speed_high)
        if in_conflict_zone:
            self.conflict_steps += 1

        self.action_change_sum += float(action_int != self.previous_action)
        self._accumulate_efficiency(speed)

        if self.variant == "standard":
            reward = self._standard_reward(speed, dist_center, exit_lane, arrived, crashed)
        else:
            reward = self._creeping_reward(speed, dist_center, exit_lane, conflict_ttc, target_speed, arrived, crashed)

        if truncated and not arrived and not crashed:
            reward -= 22.0 if self.variant == "standard" else 32.0

        step_count = max(1, int(round(self.env.unwrapped.time * self.cfg.policy_frequency)))
        info.update(
            {
                "arrived": arrived,
                "ttc": ttc,
                "conflict_ttc": conflict_ttc,
                "dist_center": dist_center,
                "ego_speed": speed,
                "exit_lane": exit_lane,
                "target_speed": target_speed,
                "in_creep_zone": in_creep_zone,
                "in_conflict_zone": in_conflict_zone,
                "speed_index": int(getattr(getattr(self.env.unwrapped, "vehicle", None), "speed_index", -1)),
                "route_action": action_int,
                "zone_steps": self.zone_steps,
                "conflict_steps": self.conflict_steps,
                "creep_zone_mean_speed": self.zone_speed_sum / max(self.zone_steps, 1),
                "high_speed_zone_rate": self.high_speed_zone_steps / max(self.zone_steps, 1),
                "creep_speed_rate": self.creep_speed_steps / max(self.zone_steps, 1),
                "mean_action_change_rate": self.action_change_sum / step_count,
                "min_ttc_seen": self.min_ttc_seen,
                "min_conflict_ttc_seen": self.min_conflict_ttc_seen,
                "ego_mean_speed": self.ego_speed_sum / step_count,
                "ego_stop_rate": self.ego_stop_steps / step_count,
                "ego_stopped_time_s": self.ego_stop_steps / self.cfg.policy_frequency,
                "ego_delay_proxy_s": self.ego_delay_sum / self.cfg.policy_frequency,
                "traffic_vehicle_seconds": self.traffic_vehicle_count_sum / self.cfg.policy_frequency,
                "traffic_mean_speed": self.traffic_speed_sum / max(self.traffic_speed_samples, 1),
                "traffic_stop_rate": self.traffic_stop_vehicle_steps / max(self.traffic_speed_samples, 1),
                "traffic_delay_proxy_s": self.traffic_delay_sum / self.cfg.policy_frequency,
                "system_delay_proxy_s": (self.ego_delay_sum + self.traffic_delay_sum) / self.cfg.policy_frequency,
                "shaped_reward": reward,
            }
        )
        self.previous_dist_center = dist_center
        self.previous_action = action_int
        return obs, float(reward), terminated, truncated, info

    def _route_progress(self, dist_center: float, exit_lane: bool) -> float:
        if math.isinf(self.previous_dist_center):
            return 0.0
        if exit_lane:
            return max(0.0, dist_center - self.previous_dist_center)
        return max(0.0, self.previous_dist_center - dist_center)

    def _standard_reward(self, speed: float, dist_center: float, exit_lane: bool, arrived: bool, crashed: bool) -> float:
        progress = self._route_progress(dist_center, exit_lane)
        reward = -0.02 + 0.10 * speed + 0.50 * progress
        if arrived:
            reward += 45.0
        if crashed:
            reward -= 40.0
        return reward

    def _creeping_reward(
        self,
        speed: float,
        dist_center: float,
        exit_lane: bool,
        ttc: float,
        target_speed: float,
        arrived: bool,
        crashed: bool,
    ) -> float:
        progress = self._route_progress(dist_center, exit_lane)
        speed_error = abs(speed - target_speed)
        speed_match = math.exp(-0.5 * (speed_error / 1.6) ** 2)
        in_creep_zone = dist_center <= self.cfg.creep_radius and not arrived
        in_conflict_zone = dist_center <= self.cfg.conflict_radius and not arrived

        reward = -0.03 + 0.04 * speed
        reward += 1.55 * progress * (0.45 + 0.55 * speed_match)

        if in_creep_zone:
            reward += (1.05 if self.cfg.interaction_reward_focus else 0.85) * speed_match
            if self.cfg.creep_speed_low <= speed <= self.cfg.creep_speed_high:
                reward += 0.45 if self.cfg.interaction_reward_focus else 0.35
            if speed > self.cfg.high_speed_zone_threshold:
                high_speed_penalty = 0.95 if self.cfg.interaction_reward_focus else 0.38
                reward -= high_speed_penalty * (speed - self.cfg.high_speed_zone_threshold)
            if self.cfg.interaction_reward_focus and speed < self.cfg.creep_speed_low:
                reward -= 0.65 * (self.cfg.creep_speed_low - speed)

        if in_conflict_zone and math.isfinite(ttc) and ttc < self.cfg.ttc_caution:
            reward += 0.25 if speed <= self.cfg.creep_speed_high else -0.55

        if math.isfinite(ttc) and ttc < self.cfg.ttc_critical:
            safety_scale = 0.35 if self.cfg.interaction_reward_focus and self.cfg.safety_shield else 1.0
            reward -= safety_scale * 4.5 * (self.cfg.ttc_critical - ttc) / self.cfg.ttc_critical
            reward -= safety_scale * 0.18 * max(speed - 1.0, 0.0)

        if speed < 1.0 and (not math.isfinite(ttc) or ttc > self.cfg.ttc_caution):
            reward -= 1.05 if self.cfg.interaction_reward_focus else 0.55
        if speed < target_speed - 1.6:
            reward -= (0.30 if self.cfg.interaction_reward_focus else 0.18) * (target_speed - speed)

        if exit_lane and dist_center > self.cfg.conflict_radius:
            reward += (0.14 if self.cfg.interaction_reward_focus else 0.08) * speed

        reward -= 0.025 * float(self.previous_action != 1)

        if arrived:
            reward += 85.0
        if crashed:
            reward -= 65.0
        return reward

    def _accumulate_efficiency(self, ego_speed: float) -> None:
        reference_speed = max(float(self.cfg.efficiency_reference_speed), 1e-6)
        stop_threshold = float(self.cfg.efficiency_stop_speed_threshold)
        self.ego_speed_sum += float(ego_speed)
        self.ego_stop_steps += int(float(ego_speed) <= stop_threshold)
        self.ego_delay_sum += max(0.0, reference_speed - float(ego_speed)) / reference_speed

        road = getattr(self.env.unwrapped, "road", None)
        ego = getattr(self.env.unwrapped, "vehicle", None)
        if road is None or ego is None:
            return
        for vehicle in getattr(road, "vehicles", []):
            if vehicle is ego:
                continue
            speed = float(getattr(vehicle, "speed", 0.0))
            self.traffic_vehicle_count_sum += 1
            self.traffic_speed_sum += speed
            self.traffic_speed_samples += 1
            self.traffic_stop_vehicle_steps += int(speed <= stop_threshold)
            self.traffic_delay_sum += max(0.0, reference_speed - speed) / reference_speed


class RouteSafetyEllipseShieldWrapper(gym.Wrapper):
    """Ego-only hard safety shield using predicted inflated-ellipse separation."""

    def __init__(self, env: gym.Env, cfg: RouteIntersectionConfig):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        raw_action = int(np.asarray(action).reshape(-1)[0])
        evaluations = {candidate: self._evaluate_candidate(candidate) for candidate in range(3)}
        raw_eval = evaluations.get(raw_action, self._empty_eval())

        progress_assisted = False
        if self.cfg.safety_progress_assist:
            progress_action = self._progress_safe_action(evaluations)
            if progress_action is not None:
                safe_action = progress_action
                progress_assisted = safe_action != raw_action
            elif raw_eval["safe"]:
                safe_action = raw_action
            else:
                safe_action = 0
        elif raw_eval["safe"]:
            safe_action = raw_action
        else:
            safe_candidates = [
                candidate
                for candidate in self._candidate_order(raw_action)
                if evaluations[candidate]["safe"]
            ]
            if safe_candidates:
                safe_action = safe_candidates[0]
            else:
                safe_action = 0

        emergency_brake_applied = False
        if (
            self.cfg.safety_emergency_brake
            and safe_action == 0
            and (safe_action != raw_action or not raw_eval["safe"])
        ):
            emergency_brake_applied = self._apply_ego_emergency_brake()

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        safe_eval = evaluations.get(safe_action, self._empty_eval())
        info = dict(info or {})
        info.update(
            {
                "safety_shield_active": True,
                "safety_raw_action": raw_action,
                "safety_action": int(safe_action),
                "safety_intervened": bool(safe_action != raw_action),
                "safety_min_h_raw": float(raw_eval["min_h"]),
                "safety_min_h_safe": float(safe_eval["min_h"]),
                "safety_min_center_distance": float(safe_eval["min_center_distance"]),
                "safety_required_distance": float(safe_eval["required_distance"]),
                "safety_candidate_safe_raw": bool(raw_eval["safe"]),
                "safety_emergency_brake_applied": bool(emergency_brake_applied),
                "safety_progress_assist": bool(self.cfg.safety_progress_assist),
                "safety_progress_assisted": bool(progress_assisted),
                "safety_neighbor_count": int(safe_eval["neighbor_count"]),
                "safety_prediction_horizon": float(self.cfg.safety_prediction_horizon),
                "safety_ellipse_margin": float(self.cfg.safety_ellipse_margin),
            }
        )
        return obs, reward, terminated, truncated, info

    def _progress_safe_action(self, evaluations: dict[int, dict]) -> int | None:
        state = ego_route_state(self.env)
        if state["dist_center"] > self.cfg.safety_progress_assist_radius:
            return None
        safe_candidates = [candidate for candidate, result in evaluations.items() if result["safe"]]
        if not safe_candidates:
            return None
        conflict_ttc = route_min_conflict_ttc(self.env)
        desired_speed = route_target_speed(
            state["dist_center"],
            conflict_ttc,
            state["exit_lane"],
            False,
            self.cfg,
        )
        if state["dist_center"] <= self.cfg.creep_radius and not state["exit_lane"]:
            desired_speed = min(desired_speed, self.cfg.creep_speed_high)

        current_speed = float(state["speed"])
        if current_speed > desired_speed + 0.8 and 0 in safe_candidates:
            return 0

        ranked = sorted(
            safe_candidates,
            key=lambda candidate: self._candidate_target_speed(getattr(self.env.unwrapped, "vehicle", None), candidate),
            reverse=True,
        )
        for candidate in ranked:
            if self._candidate_target_speed(getattr(self.env.unwrapped, "vehicle", None), candidate) <= desired_speed + 0.25:
                return int(candidate)
        return min(safe_candidates)

    def _apply_ego_emergency_brake(self) -> bool:
        ego = getattr(self.env.unwrapped, "vehicle", None)
        if ego is None or not hasattr(ego, "speed"):
            return False
        speed = float(getattr(ego, "speed", 0.0))
        decel_step = float(self.cfg.safety_emergency_decel) / max(float(self.cfg.policy_frequency), 1.0)
        new_speed = max(0.0, speed - decel_step)
        try:
            ego.speed = new_speed
        except Exception:
            return False
        if hasattr(ego, "target_speed"):
            try:
                ego.target_speed = min(float(getattr(ego, "target_speed", new_speed)), new_speed)
            except Exception:
                pass
        if hasattr(ego, "speed_index"):
            try:
                ego.speed_index = 0
            except Exception:
                pass
        return new_speed < speed - 1e-6

    def _candidate_order(self, raw_action: int) -> list[int]:
        ordered = [raw_action]
        for candidate in self.cfg.safety_action_preference:
            if int(candidate) not in ordered and 0 <= int(candidate) <= 2:
                ordered.append(int(candidate))
        for candidate in (0, 1, 2):
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered

    @staticmethod
    def _empty_eval() -> dict:
        return {
            "safe": True,
            "min_h": math.inf,
            "min_center_distance": math.inf,
            "required_distance": 0.0,
            "neighbor_count": 0,
        }

    def _candidate_target_speed(self, ego, action: int) -> float:
        speeds = list(self.cfg.target_speeds)
        if not speeds:
            return float(getattr(ego, "speed", 0.0))
        current_speed = float(getattr(ego, "speed", 0.0))
        speed_index = getattr(ego, "speed_index", None)
        if speed_index is None:
            speed_index = int(np.argmin(np.abs(np.asarray(speeds, dtype=np.float32) - current_speed)))
        speed_index = int(np.clip(speed_index, 0, len(speeds) - 1))
        if action == 0:
            speed_index -= 1
        elif action == 2:
            speed_index += 1
        speed_index = int(np.clip(speed_index, 0, len(speeds) - 1))
        return float(speeds[speed_index])

    def _predict_ego_position(self, ego, action: int, t: float) -> np.ndarray:
        position = np.asarray(ego.position, dtype=np.float32)
        speed = float(getattr(ego, "speed", 0.0))
        heading = float(getattr(ego, "heading", 0.0))
        target_speed = self._candidate_target_speed(ego, action)
        max_delta = float(self.cfg.safety_max_accel) * float(t)
        future_speed = speed + float(np.clip(target_speed - speed, -max_delta, max_delta))
        mean_speed = 0.5 * (speed + future_speed)
        direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32)
        return position + direction * mean_speed * float(t)

    def _evaluate_candidate(self, action: int) -> dict:
        ego = getattr(self.env.unwrapped, "vehicle", None)
        road = getattr(self.env.unwrapped, "road", None)
        if ego is None or road is None:
            return self._empty_eval()

        ego_position = np.asarray(ego.position, dtype=np.float32)
        ego_dist_center_now = float(np.linalg.norm(ego_position))
        if ego_dist_center_now > self.cfg.safety_apply_radius:
            return self._empty_eval()
        ego_heading = float(getattr(ego, "heading", 0.0))
        ego_length = float(getattr(ego, "LENGTH", getattr(ego, "length", 5.0)))
        ego_width = float(getattr(ego, "WIDTH", getattr(ego, "width", 2.0)))

        min_h = math.inf
        min_center_distance = math.inf
        required_at_min = 0.0
        neighbor_count = 0
        horizon = max(float(self.cfg.safety_prediction_horizon), float(self.cfg.safety_prediction_dt))
        dt = max(float(self.cfg.safety_prediction_dt), 1e-3)
        times = np.arange(0.0, horizon + 0.5 * dt, dt, dtype=np.float32)

        for other in road.vehicles:
            if other is ego:
                continue
            other_position = np.asarray(other.position, dtype=np.float32)
            if float(np.linalg.norm(other_position - ego_position)) > self.cfg.safety_neighbor_range:
                continue
            neighbor_count += 1
            other_velocity = _vehicle_velocity(other)
            other_heading = float(getattr(other, "heading", 0.0))
            other_length = float(getattr(other, "LENGTH", getattr(other, "length", 5.0)))
            other_width = float(getattr(other, "WIDTH", getattr(other, "width", 2.0)))
            current_pair_distance = float(np.linalg.norm(other_position - ego_position))
            ego_conflict_times: list[float] = []
            other_conflict_times: list[float] = []

            for t in times:
                t_float = float(t)
                future_ego = self._predict_ego_position(ego, action, t_float)
                future_other = other_position + other_velocity * t_float
                ego_conflict = float(np.linalg.norm(future_ego)) <= self.cfg.safety_conflict_radius
                other_conflict = float(np.linalg.norm(future_other)) <= self.cfg.safety_conflict_radius
                if ego_conflict:
                    ego_conflict_times.append(t_float)
                if other_conflict:
                    other_conflict_times.append(t_float)
                near_field = (
                    current_pair_distance <= self.cfg.safety_near_field_radius
                    and t_float <= self.cfg.safety_near_field_horizon
                )
                if not ((ego_conflict and other_conflict) or near_field):
                    continue
                h, center_distance, required_distance = vehicle_ellipse_clearance(
                    future_ego,
                    ego_heading,
                    ego_length,
                    ego_width,
                    future_other,
                    other_heading,
                    other_length,
                    other_width,
                    self.cfg.safety_ellipse_margin,
                )
                if h < min_h:
                    min_h = h
                    min_center_distance = center_distance
                    required_at_min = required_distance

            if self.cfg.safety_time_gate and ego_conflict_times and other_conflict_times:
                ego_start, ego_end = min(ego_conflict_times), max(ego_conflict_times)
                other_start, other_end = min(other_conflict_times), max(other_conflict_times)
                buffer = float(self.cfg.safety_time_gate_buffer)
                intervals_overlap = ego_start <= other_end + buffer and other_start <= ego_end + buffer
                if intervals_overlap and min_h > -1.0:
                    min_h = -1.0
                    min_center_distance = min(min_center_distance, float(np.linalg.norm(other_position - ego_position)))
                    required_at_min = max(required_at_min, 2.0 * self.cfg.safety_conflict_radius)

        if not math.isfinite(min_h):
            return self._empty_eval()

        return {
            "safe": bool(min_h >= self.cfg.safety_min_clearance),
            "min_h": float(min_h),
            "min_center_distance": float(min_center_distance),
            "required_distance": float(required_at_min),
            "neighbor_count": int(neighbor_count),
        }


def make_route_intersection_env(
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    seed: int | None = None,
    render_mode: str | None = None,
    use_attention_obs: bool = False,
) -> gym.Env:
    env = gym.make("intersection-v0", render_mode=render_mode)
    env.unwrapped.configure(highway_route_env_config(cfg))
    env = RouteCreepingRewardWrapper(env, variant=variant, cfg=cfg)
    if cfg.safety_shield:
        env = RouteSafetyEllipseShieldWrapper(env, cfg=cfg)
    if use_attention_obs:
        env = RouteAttentionObservationWrapper(env, cfg=cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_route_vec_env(
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    n_envs: int,
    seed: int,
    backend: VecBackend = "dummy",
    use_attention_obs: bool = False,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    def thunk(rank: int):
        def _init():
            return make_route_intersection_env(
                variant=variant,
                cfg=cfg,
                seed=seed + rank,
                use_attention_obs=use_attention_obs,
            )

        return _init

    env_fns = [thunk(rank) for rank in range(n_envs)]
    if backend == "subproc":
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    return VecMonitor(vec_env)


class NearVehicleAttentionExtractor:
    """Factory-compatible SB3 feature extractor with distance-biased attention."""

    def __new__(
        cls,
        observation_space,
        features_dim: int = 128,
        vehicles_count: int = ROUTE_VEHICLE_COUNT,
        token_features: int = ATTENTION_TOKEN_FEATURES,
        aux_features: int = ATTENTION_AUX_FEATURES,
        embed_dim: int = 64,
        distance_bias: float = 2.5,
    ):
        import torch
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class _NearVehicleAttentionExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space):
                super().__init__(observation_space, features_dim)
                self.vehicles_count = vehicles_count
                self.token_features = token_features
                self.aux_features = aux_features
                self.token_size = vehicles_count * token_features
                self.distance_bias = float(distance_bias)
                self.ego_encoder = nn.Sequential(
                    nn.Linear(token_features, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                )
                self.vehicle_encoder = nn.Sequential(
                    nn.Linear(token_features, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                )
                self.query = nn.Linear(embed_dim, embed_dim)
                self.key = nn.Linear(embed_dim, embed_dim)
                self.value = nn.Linear(embed_dim, embed_dim)
                self.head = nn.Sequential(
                    nn.Linear(embed_dim * 2 + aux_features, features_dim),
                    nn.ReLU(),
                    nn.Linear(features_dim, features_dim),
                    nn.ReLU(),
                )

            def forward(self, observations):
                x = observations.float()
                tokens = x[:, : self.token_size].reshape(-1, self.vehicles_count, self.token_features)
                aux = x[:, self.token_size : self.token_size + self.aux_features]
                ego_token = tokens[:, 0, :]
                other_tokens = tokens[:, 1:, :]
                mask = other_tokens[:, :, 0] > 0.5
                distances = other_tokens[:, :, 7].clamp(0.0, 2.0)

                ego_embedding = self.ego_encoder(ego_token)
                vehicle_embedding = self.vehicle_encoder(other_tokens)
                query = self.query(ego_embedding).unsqueeze(1)
                keys = self.key(vehicle_embedding)
                values = self.value(vehicle_embedding)

                scores = (query * keys).sum(dim=-1) / math.sqrt(keys.shape[-1])
                scores = scores - self.distance_bias * distances
                scores = scores.masked_fill(~mask, -1e9)
                weights = torch.softmax(scores, dim=1) * mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                context = (weights.unsqueeze(-1) * values).sum(dim=1)
                return self.head(torch.cat([ego_embedding, context, aux], dim=1))

        return _NearVehicleAttentionExtractor(observation_space)


def train_route_ppo(
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    total_timesteps: int,
    model_path: Path,
    n_envs: int = 16,
    seed: int = 7,
    device: str = "auto",
    backend: VecBackend = "dummy",
    use_attention_obs: bool = False,
    learning_rate: float = 2.5e-4,
):
    from stable_baselines3 import PPO

    env = make_route_vec_env(
        variant=variant,
        cfg=cfg,
        n_envs=n_envs,
        seed=seed,
        backend=backend,
        use_attention_obs=use_attention_obs,
    )
    n_steps = 128
    rollout_size = n_steps * n_envs
    policy_kwargs = {}
    if use_attention_obs:
        policy_kwargs = {
            "features_extractor_class": NearVehicleAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128,
                "vehicles_count": cfg.vehicles_count,
                "token_features": ATTENTION_TOKEN_FEATURES,
                "aux_features": ATTENTION_AUX_FEATURES,
            },
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
        }
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=min(512, rollout_size),
        n_epochs=4,
        gamma=0.985,
        gae_lambda=0.92,
        learning_rate=learning_rate,
        ent_coef=0.015,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def finetune_route_ppo(
    base_model_path: Path,
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    total_timesteps: int,
    model_path: Path,
    n_envs: int = 4,
    seed: int = 7,
    device: str = "auto",
    backend: VecBackend = "dummy",
    use_attention_obs: bool = False,
    learning_rate: float = 1.0e-4,
):
    from stable_baselines3 import PPO

    env = make_route_vec_env(
        variant=variant,
        cfg=cfg,
        n_envs=n_envs,
        seed=seed,
        backend=backend,
        use_attention_obs=use_attention_obs,
    )
    model = PPO.load(str(base_model_path), env=env, device=device)
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    model.ent_coef = 0.01
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model


def route_attention_policy_kwargs(cfg: RouteIntersectionConfig) -> dict:
    return {
        "features_extractor_class": NearVehicleAttentionExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "vehicles_count": cfg.vehicles_count,
            "token_features": ATTENTION_TOKEN_FEATURES,
            "aux_features": ATTENTION_AUX_FEATURES,
        },
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }


def collect_route_teacher_dataset(
    teacher,
    cfg: RouteIntersectionConfig,
    episodes: int = 120,
    seed: int = 14000,
    variant: RouteVariant = "creeping",
    use_attention_obs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []
    for episode in range(episodes):
        env = make_route_intersection_env(
            variant=variant,
            cfg=cfg,
            seed=seed + episode,
            use_attention_obs=use_attention_obs,
        )
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action, _ = teacher.predict(obs, deterministic=True)
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(int(np.asarray(action).reshape(-1)[0]))
            obs, _, terminated, truncated, _ = env.step(actions[-1])
            done = bool(terminated or truncated)
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def behavior_clone_route_ppo_actor(
    cfg: RouteIntersectionConfig,
    model_path: Path,
    base_model_path: Path | None = None,
    teacher=None,
    episodes: int = 160,
    seed: int = 14000,
    epochs: int = 35,
    batch_size: int = 1024,
    learning_rate: float = 7e-4,
    device: str = "cpu",
    use_attention_obs: bool = True,
):
    import torch
    from stable_baselines3 import PPO

    teacher = teacher or RouteScriptedCreepingPolicy()
    obs_np, act_np = collect_route_teacher_dataset(
        teacher,
        cfg=cfg,
        episodes=episodes,
        seed=seed,
        use_attention_obs=use_attention_obs,
    )

    if base_model_path is not None:
        model = PPO.load(str(base_model_path), device=device)
    else:
        env = make_route_vec_env(
            variant="creeping",
            cfg=cfg,
            n_envs=1,
            seed=seed,
            backend="dummy",
            use_attention_obs=use_attention_obs,
        )
        policy_kwargs = route_attention_policy_kwargs(cfg) if use_attention_obs else {}
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=128,
            n_epochs=1,
            gamma=0.985,
            gae_lambda=0.92,
            learning_rate=learning_rate,
            ent_coef=0.0,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )
        env.close()

    policy = model.policy
    policy.train()
    params = list(policy.features_extractor.parameters())
    params += list(policy.mlp_extractor.policy_net.parameters())
    params += list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(act_np, dtype=torch.long, device=model.device)
    n_samples = obs_t.shape[0]

    losses = []
    accuracies = []
    for _ in range(epochs):
        permutation = torch.randperm(n_samples, device=model.device)
        epoch_losses = []
        epoch_acc = []
        for start in range(0, n_samples, batch_size):
            idx = permutation[start : start + batch_size]
            distribution = policy.get_distribution(obs_t[idx])
            log_prob = distribution.log_prob(act_t[idx])
            entropy = distribution.entropy().mean()
            loss = -log_prob.mean() - 0.001 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                predicted = distribution.distribution.probs.argmax(dim=1)
                epoch_acc.append(float((predicted == act_t[idx]).float().mean().detach().cpu()))
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        accuracies.append(float(np.mean(epoch_acc)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(losses) + 1),
            "bc_loss": losses,
            "bc_action_accuracy": accuracies,
            "samples": n_samples,
        }
    )
    return model, history


def evaluate_route_agent(
    model,
    agent_name: str,
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    episodes: int = 50,
    seed: int = 1000,
    deterministic: bool = True,
    use_attention_obs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for episode in range(episodes):
        env = make_route_intersection_env(
            variant=variant,
            cfg=cfg,
            seed=seed + episode,
            use_attention_obs=use_attention_obs,
        )
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0
        total_reward = 0.0
        collided = False
        arrived = False
        final_info = {}
        action_counts = {0: 0, 1: 0, 2: 0}
        safety_interventions = 0
        safety_violation_steps = 0
        safety_min_h = math.inf
        safety_neighbor_steps = 0
        safety_emergency_brake_steps = 0
        safety_progress_assist_steps = 0
        max_steps = int(math.ceil(float(cfg.duration) * float(cfg.policy_frequency))) + 5

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(np.asarray(action).reshape(-1)[0])
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)
            collided = collided or bool(info.get("crashed", False))
            arrived = arrived or bool(info.get("arrived", False))
            safety_interventions += int(bool(info.get("safety_intervened", False)))
            safety_emergency_brake_steps += int(bool(info.get("safety_emergency_brake_applied", False)))
            safety_progress_assist_steps += int(bool(info.get("safety_progress_assisted", False)))
            safe_h = float(info.get("safety_min_h_safe", math.inf))
            if math.isfinite(safe_h):
                safety_min_h = min(safety_min_h, safe_h)
                safety_violation_steps += int(safe_h < cfg.safety_min_clearance)
            safety_neighbor_steps += int(info.get("safety_neighbor_count", 0) > 0)
            final_info = info
        if not done and steps >= max_steps:
            final_info = dict(final_info or {})
            final_info["max_step_guard"] = True

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
                "min_ttc": final_info.get("min_ttc_seen", np.nan),
                "min_conflict_ttc": final_info.get("min_conflict_ttc_seen", np.nan),
                "creep_zone_mean_speed": final_info.get("creep_zone_mean_speed", np.nan),
                "high_speed_zone_rate": final_info.get("high_speed_zone_rate", np.nan),
                "creep_speed_rate": final_info.get("creep_speed_rate", np.nan),
                "conflict_steps": final_info.get("conflict_steps", np.nan),
                "mean_action_change_rate": final_info.get("mean_action_change_rate", np.nan),
                "ego_mean_speed": final_info.get("ego_mean_speed", np.nan),
                "ego_stop_rate": final_info.get("ego_stop_rate", np.nan),
                "ego_stopped_time_s": final_info.get("ego_stopped_time_s", np.nan),
                "ego_delay_proxy_s": final_info.get("ego_delay_proxy_s", np.nan),
                "traffic_vehicle_seconds": final_info.get("traffic_vehicle_seconds", np.nan),
                "traffic_mean_speed": final_info.get("traffic_mean_speed", np.nan),
                "traffic_stop_rate": final_info.get("traffic_stop_rate", np.nan),
                "traffic_delay_proxy_s": final_info.get("traffic_delay_proxy_s", np.nan),
                "system_delay_proxy_s": final_info.get("system_delay_proxy_s", np.nan),
                "slower_action_rate": action_counts.get(0, 0) / max(steps, 1),
                "idle_action_rate": action_counts.get(1, 0) / max(steps, 1),
                "faster_action_rate": action_counts.get(2, 0) / max(steps, 1),
                "safety_shield": bool(cfg.safety_shield),
                "safety_intervention_rate": safety_interventions / max(steps, 1),
                "safety_emergency_brake_rate": safety_emergency_brake_steps / max(steps, 1),
                "safety_progress_assist_rate": safety_progress_assist_steps / max(steps, 1),
                "safety_violation_rate": safety_violation_steps / max(steps, 1),
                "safety_min_h": safety_min_h if math.isfinite(safety_min_h) else np.nan,
                "safety_neighbor_rate": safety_neighbor_steps / max(steps, 1),
            }
        )
        env.close()

    df = pd.DataFrame(rows)
    return df, summarize_route_metrics(df)


def summarize_route_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (maneuver, agent), group in df.groupby(["maneuver", "agent"], dropna=False):
        collided = group["collided"].astype(bool)
        collision_times = group.loc[collided, "time_to_collision_s"].dropna()
        rows.append(
            {
                "maneuver": maneuver,
                "agent": agent,
                "episodes": len(group),
                "success_rate": group["success"].mean(),
                "collision_rate": collided.mean(),
                "timeout_rate": group["timeout"].mean(),
                "mean_return": group["return"].mean(),
                "mean_survival_time_s": group["survival_time_s"].mean(),
                "mean_time_to_collision_s": collision_times.mean() if len(collision_times) else np.nan,
                "mean_creep_zone_speed": group["creep_zone_mean_speed"].mean(),
                "high_speed_zone_rate": group["high_speed_zone_rate"].mean(),
                "creep_speed_rate": group["creep_speed_rate"].mean(),
                "mean_min_ttc": group["min_ttc"].replace(np.inf, np.nan).mean(),
                "mean_min_conflict_ttc": group["min_conflict_ttc"].replace(np.inf, np.nan).mean(),
                "mean_action_change_rate": group["mean_action_change_rate"].mean(),
                "mean_ego_speed": group["ego_mean_speed"].mean() if "ego_mean_speed" in group else np.nan,
                "ego_stop_rate": group["ego_stop_rate"].mean() if "ego_stop_rate" in group else np.nan,
                "mean_ego_stopped_time_s": group["ego_stopped_time_s"].mean()
                if "ego_stopped_time_s" in group
                else np.nan,
                "mean_ego_delay_proxy_s": group["ego_delay_proxy_s"].mean()
                if "ego_delay_proxy_s" in group
                else np.nan,
                "mean_traffic_vehicle_seconds": group["traffic_vehicle_seconds"].mean()
                if "traffic_vehicle_seconds" in group
                else np.nan,
                "mean_traffic_speed": group["traffic_mean_speed"].mean()
                if "traffic_mean_speed" in group
                else np.nan,
                "traffic_stop_rate": group["traffic_stop_rate"].mean()
                if "traffic_stop_rate" in group
                else np.nan,
                "mean_traffic_delay_proxy_s": group["traffic_delay_proxy_s"].mean()
                if "traffic_delay_proxy_s" in group
                else np.nan,
                "mean_system_delay_proxy_s": group["system_delay_proxy_s"].mean()
                if "system_delay_proxy_s" in group
                else np.nan,
                "slower_action_rate": group["slower_action_rate"].mean(),
                "idle_action_rate": group["idle_action_rate"].mean(),
                "faster_action_rate": group["faster_action_rate"].mean(),
                "safety_shield": bool(group["safety_shield"].astype(bool).any())
                if "safety_shield" in group
                else False,
                "mean_safety_intervention_rate": group["safety_intervention_rate"].mean()
                if "safety_intervention_rate" in group
                else np.nan,
                "mean_safety_emergency_brake_rate": group["safety_emergency_brake_rate"].mean()
                if "safety_emergency_brake_rate" in group
                else np.nan,
                "mean_safety_progress_assist_rate": group["safety_progress_assist_rate"].mean()
                if "safety_progress_assist_rate" in group
                else np.nan,
                "mean_safety_violation_rate": group["safety_violation_rate"].mean()
                if "safety_violation_rate" in group
                else np.nan,
                "mean_safety_min_h": group["safety_min_h"].replace(np.inf, np.nan).mean()
                if "safety_min_h" in group
                else np.nan,
                "mean_safety_neighbor_rate": group["safety_neighbor_rate"].mean()
                if "safety_neighbor_rate" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


class RouteFastPolicy:
    def predict(self, obs, deterministic: bool = True):
        return 2, None


class RouteHoldPolicy:
    def predict(self, obs, deterministic: bool = True):
        return 1, None


class RouteScriptedCreepingPolicy:
    """Diagnostic policy only. It is not used as an action constraint for PPO."""

    def __init__(
        self,
        clear_speed: float = 4.8,
        caution_speed: float = 2.8,
        approach_clear_speed: float = 6.4,
        approach_caution_speed: float = 3.8,
        caution_ttc: float = 3.5,
    ):
        self.clear_speed = float(clear_speed)
        self.caution_speed = float(caution_speed)
        self.approach_clear_speed = float(approach_clear_speed)
        self.approach_caution_speed = float(approach_caution_speed)
        self.caution_ttc = float(caution_ttc)

    def predict(self, obs, deterministic: bool = True):
        raw = np.asarray(obs, dtype=np.float32)
        if raw.ndim == 1 and raw.size == ATTENTION_OBS_SIZE:
            ego_speed = float(raw[-6]) * 9.0
            dist_center = float(raw[-5]) * 60.0
            ttc_feature = float(raw[-4])
            ttc = math.inf if ttc_feature >= 0.999 else 10.0 * ttc_feature
        else:
            kin = raw.reshape(ROUTE_VEHICLE_COUNT, ROUTE_RAW_FEATURES)
            ego = kin[0]
            ego_speed = float(np.linalg.norm(ego[3:5]))
            dist_center = float(np.linalg.norm(ego[1:3]))
            ttc = raw_observation_conflict_ttc(kin)

        if dist_center <= 18.0 and (not math.isfinite(ttc) or ttc > 3.0):
            target_speed = self.clear_speed
        elif dist_center <= 24.0:
            target_speed = self.caution_speed if math.isfinite(ttc) and ttc < self.caution_ttc else self.clear_speed
        elif dist_center <= 36.0:
            target_speed = self.approach_caution_speed if math.isfinite(ttc) and ttc < self.caution_ttc else self.approach_clear_speed
        else:
            target_speed = 8.5

        if ego_speed > target_speed + 0.8:
            return 0, None
        if ego_speed < target_speed - 0.8:
            return 2, None
        return 1, None


class RouteLeftCautiousCreepingPolicy(RouteScriptedCreepingPolicy):
    """Cautious left-turn teacher used for curriculum pretraining only."""

    def __init__(self):
        super().__init__(
            clear_speed=3.8,
            caution_speed=1.5,
            approach_clear_speed=5.6,
            approach_caution_speed=2.8,
            caution_ttc=4.5,
        )


def evaluate_route_reference_policies(
    cfg: RouteIntersectionConfig,
    episodes: int = 50,
    seed: int = 3000,
    use_attention_obs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs: list[tuple[str, object, RouteVariant]] = [
        ("scripted_fast", RouteFastPolicy(), "standard"),
        ("scripted_hold_speed", RouteHoldPolicy(), "standard"),
        ("scripted_reward_creeping", RouteScriptedCreepingPolicy(), "creeping"),
    ]
    frames = []
    summaries = []
    for name, policy, variant in specs:
        df, summary = evaluate_route_agent(
            policy,
            name,
            variant=variant,
            cfg=cfg,
            episodes=episodes,
            seed=seed,
            use_attention_obs=use_attention_obs,
        )
        frames.append(df)
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.concat(summaries, ignore_index=True)


def load_route_ppo(model_path: Path, device: str = "auto"):
    from stable_baselines3 import PPO

    return PPO.load(str(model_path), device=device)


def render_route_episodes(
    model,
    agent_name: str,
    variant: RouteVariant,
    cfg: RouteIntersectionConfig,
    video_dir: Path,
    episodes: int = 10,
    seed: int = 9000,
    deterministic: bool = True,
    use_attention_obs: bool = False,
) -> pd.DataFrame:
    import imageio.v2 as imageio

    rows = []
    video_dir = video_dir.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(episodes):
        env = make_route_intersection_env(
            variant=variant,
            cfg=cfg,
            seed=seed + episode,
            render_mode="rgb_array",
            use_attention_obs=use_attention_obs,
        )
        video_path = video_dir / f"{agent_name}_{cfg.maneuver}_ep{episode:02d}.mp4"
        writer = imageio.get_writer(video_path, fps=cfg.policy_frequency)
        obs, _ = env.reset(seed=seed + episode)
        frame = env.render()
        if frame is not None:
            writer.append_data(np.asarray(frame))
        done = False
        total_reward = 0.0
        steps = 0
        collided = False
        arrived = False
        final_info = {}
        safety_interventions = 0
        safety_violation_steps = 0
        safety_min_h = math.inf
        safety_emergency_brake_steps = 0
        safety_progress_assist_steps = 0
        max_steps = int(math.ceil(float(cfg.duration) * float(cfg.policy_frequency))) + 5
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).reshape(-1)[0]))
            frame = env.render()
            if frame is not None:
                writer.append_data(np.asarray(frame))
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
            collided = collided or bool(info.get("crashed", False))
            arrived = arrived or bool(info.get("arrived", False))
            safety_interventions += int(bool(info.get("safety_intervened", False)))
            safety_emergency_brake_steps += int(bool(info.get("safety_emergency_brake_applied", False)))
            safety_progress_assist_steps += int(bool(info.get("safety_progress_assisted", False)))
            safe_h = float(info.get("safety_min_h_safe", math.inf))
            if math.isfinite(safe_h):
                safety_min_h = min(safety_min_h, safe_h)
                safety_violation_steps += int(safe_h < cfg.safety_min_clearance)
            final_info = info
        if not done and steps >= max_steps:
            final_info = dict(final_info or {})
            final_info["max_step_guard"] = True
        rows.append(
            {
                "maneuver": cfg.maneuver,
                "agent": agent_name,
                "episode": episode,
                "success": bool(arrived and not collided),
                "collided": collided,
                "return": total_reward,
                "steps": steps,
                "survival_time_s": steps / cfg.policy_frequency,
                "creep_zone_mean_speed": final_info.get("creep_zone_mean_speed", np.nan),
                "high_speed_zone_rate": final_info.get("high_speed_zone_rate", np.nan),
                "creep_speed_rate": final_info.get("creep_speed_rate", np.nan),
                "min_ttc": final_info.get("min_ttc_seen", np.nan),
                "min_conflict_ttc": final_info.get("min_conflict_ttc_seen", np.nan),
                "ego_mean_speed": final_info.get("ego_mean_speed", np.nan),
                "ego_stop_rate": final_info.get("ego_stop_rate", np.nan),
                "ego_stopped_time_s": final_info.get("ego_stopped_time_s", np.nan),
                "ego_delay_proxy_s": final_info.get("ego_delay_proxy_s", np.nan),
                "traffic_vehicle_seconds": final_info.get("traffic_vehicle_seconds", np.nan),
                "traffic_mean_speed": final_info.get("traffic_mean_speed", np.nan),
                "traffic_stop_rate": final_info.get("traffic_stop_rate", np.nan),
                "traffic_delay_proxy_s": final_info.get("traffic_delay_proxy_s", np.nan),
                "system_delay_proxy_s": final_info.get("system_delay_proxy_s", np.nan),
                "safety_shield": bool(cfg.safety_shield),
                "safety_intervention_rate": safety_interventions / max(steps, 1),
                "safety_emergency_brake_rate": safety_emergency_brake_steps / max(steps, 1),
                "safety_progress_assist_rate": safety_progress_assist_steps / max(steps, 1),
                "safety_violation_rate": safety_violation_steps / max(steps, 1),
                "safety_min_h": safety_min_h if math.isfinite(safety_min_h) else np.nan,
            }
        )
        writer.close()
        env.close()
    metrics = pd.DataFrame(rows)
    video_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = video_dir / f"{agent_name}_{cfg.maneuver}_rendered_episode_metrics.csv"
    try:
        metrics.to_csv(metrics_path, index=False)
    except OSError:
        try:
            fallback_dir = Path.cwd() / "notebooks" / "results" / "render_metrics"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / f"{video_dir.name}_{agent_name}_{cfg.maneuver}_rendered_episode_metrics.csv"
            metrics.to_csv(fallback_path, index=False)
        except OSError:
            pass
    return metrics
