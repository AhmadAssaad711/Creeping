"""The learned creeping policy as a standalone driving skill.

``LearnedCreepingSkill`` runs a frozen route-attention PPO creeping checkpoint against any
``intersection-v0``-geometry environment (the junction centre at the world origin, the ego's
route ending at ``o1``/``o2``/``o3``) and turns its discrete longitudinal action into a speed
request. It is the black box behind the orchestrator's ``LearnedCreepingWrapper``
(``av-orchestrator/cso/wrappers/learned_creeping.py``); the wrapper never sees the network,
the observation schema or the checkpoint path.

Observation parity. The policy was trained on an absolute, unnormalised, distance-sorted
15 x 7 Kinematics frame (:func:`creeping.route_attention.highway_route_env_config`) passed
through :func:`creeping.route_attention.route_attention_features`. The skill builds exactly that
frame with its own ``KinematicObservation`` on the host env, whatever observation the host was
configured with. ``order="sorted"`` draws nothing from ``np_random``, so querying the skill
never changes the host's traffic.

Action semantics. Training used ``DiscreteMetaAction`` (longitudinal only, SLOWER/IDLE/FASTER)
over :attr:`RouteIntersectionConfig.target_speeds`. The skill maps the chosen action to the
neighbouring grid speed of the ego's *current* speed, so the request is meaningful on a host
with a different speed grid or policy frequency. A host at another policy frequency is outside
the training distribution; that is a calibration question for the host, not hidden here.

Only straight (``o2``) and left (``o1``) checkpoints exist; on a right turn (``o3``) or with no
resolvable route the skill makes no request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from creeping.paths import ROUTE_ATTENTION_RESULTS
from creeping.route_attention import (
    Maneuver,
    ego_route_state,
    has_arrived,
    highway_route_env_config,
    load_route_ppo,
    route_attention_features,
    route_config_for,
)

DEFAULT_MODEL_DIR = ROUTE_ATTENTION_RESULTS / "models"
DEFAULT_MODEL_FAMILY = "ppo_attention_bc_creeping_calibrated_v2"

#: DiscreteMetaAction longitudinal-only indexes (highway_env ACTIONS_LONGI).
ACTION_SPEED_DELTA = {0: -1, 1: 0, 2: +1}

DESTINATION_MANEUVER: dict[str, Maneuver] = {"o2": "straight", "o1": "left"}


@lru_cache(maxsize=None)
def _load_policy(model_path: str, device: str):
    return load_route_ppo(Path(model_path), device=device)


def route_destination(env) -> Optional[str]:
    """The exit node (``o1``/``o2``/``o3``) of the ego's planned route, or ``None``."""
    ego = getattr(env.unwrapped, "vehicle", None)
    route = getattr(ego, "route", None) or []
    for lane_index in reversed(route):
        if isinstance(lane_index, tuple) and str(lane_index[1]).startswith("o"):
            return str(lane_index[1])
    destination = env.unwrapped.config.get("destination")
    return str(destination) if destination else None


class LearnedCreepingSkill:
    """Frozen route-attention creeping policy exposed as a speed request.

    Args:
        model_dir: folder holding ``{maneuver}_{model_family}.zip`` checkpoints.
        model_family: checkpoint family; the default is the calibrated behaviour-cloned
            creeping policy used by the 12/0.5 benchmark ("learned creeping").
        device: torch device for inference.
        **cfg_overrides: forwarded to :func:`route_config_for` (radii, speed grid, ...); they
            must match what the checkpoint was trained with.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        model_family: str = DEFAULT_MODEL_FAMILY,
        device: str = "cpu",
        **cfg_overrides,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_family = model_family
        self.device = device
        self.cfg_overrides = cfg_overrides
        self._observers: dict[int, object] = {}

    # ---------------------------------------------------------------------------------- #
    # Pure queries
    # ---------------------------------------------------------------------------------- #
    def maneuver(self, env) -> Optional[Maneuver]:
        return DESTINATION_MANEUVER.get(route_destination(env) or "")

    def config(self, env):
        maneuver = self.maneuver(env)
        return None if maneuver is None else route_config_for(maneuver, **self.cfg_overrides)

    def model_path(self, maneuver: Maneuver) -> Path:
        return self.model_dir / f"{maneuver}_{self.model_family}.zip"

    def raw_frame(self, env, cfg) -> np.ndarray:
        """The 15 x 7 absolute Kinematics frame the policy was trained on, built on ``env``."""
        from highway_env.envs.common.observation import observation_factory

        host = env.unwrapped
        observer = self._observers.get(id(host))
        if observer is None or observer.env is not host:
            observer = observation_factory(host, highway_route_env_config(cfg)["observation"])
            self._observers[id(host)] = observer
        return np.asarray(observer.observe(), dtype=np.float32)

    def observation(self, env) -> Optional[np.ndarray]:
        cfg = self.config(env)
        if cfg is None:
            return None
        return route_attention_features(self.raw_frame(env, cfg), env, cfg)

    def route_state(self, env) -> dict:
        state = ego_route_state(env)
        state["arrived"] = has_arrived(env, {})
        return state

    def in_creep_zone(self, env) -> bool:
        """Within the trained approach radius of the junction and not yet on an exit lane."""
        cfg = self.config(env)
        if cfg is None:
            return False
        state = self.route_state(env)
        return bool(state["dist_center"] <= cfg.approach_radius and not state["exit_lane"] and not state["arrived"])

    def get_action(self, env) -> Optional[int]:
        """The policy's deterministic SLOWER/IDLE/FASTER index, or None off-route."""
        maneuver = self.maneuver(env)
        obs = self.observation(env)
        if maneuver is None or obs is None:
            return None
        policy = _load_policy(str(self.model_path(maneuver)), self.device)
        action, _ = policy.predict(obs, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    def get_speed_request(self, env) -> Optional[float]:
        """Speed the policy asks for this step, on its training speed grid, or None when the
        ego is outside the creeping zone or on an untrained route."""
        if not self.in_creep_zone(env):
            return None
        action = self.get_action(env)
        if action is None:
            return None
        grid = np.asarray(self.config(env).target_speeds, dtype=float)
        speed = float(env.unwrapped.vehicle.speed)
        index = int(np.argmin(np.abs(grid - speed)))
        index = int(np.clip(index + ACTION_SPEED_DELTA.get(action, 0), 0, len(grid) - 1))
        return float(grid[index])
