"""
simulation/environment.py
--------------------------
SumoEnv: T-intersection creeping negotiation environment.

Observation vector (18 dims)
------------------------------
 [0]  dist_to_j1_norm          ego distance to J1 / approach_length   [0=at J1, 1=far]
 [1]  ego_speed_norm            ego speed / MAX_SPEED
 [2]  ttc_norm                  TTC / max_ttc
 [3]  in_creep_zone             1.0 if within creep_zone_dist of J1

 [4]  cv1_present               1 if nearest crossing vehicle present
 [5]  cv1_tta_norm              time-to-reach-J1 / 10s
 [6]  cv1_speed_norm            cv1 speed / MAX_SPEED
 [7]  cv1_accel_norm            cv1 accel / 4.0  (negative = braking = yielding)

 [8]  cv2_present
 [9]  cv2_tta_norm
[10]  cv2_speed_norm
[11]  cv2_accel_norm

[12]  gap1_size_norm            best upcoming gap / 10s
[13]  gap1_tta_norm             time until gap1 opens / 10s
[14]  gap2_size_norm
[15]  gap2_tta_norm

[16]  approach_speed_excess     (ego_speed - target_speed_for_distance) / MAX_SPEED
                                 > 0 means going too fast for current position
                                 This directly tells the agent to slow down
[17]  time_in_zone_norm         steps_in_creep_zone / timeout_steps

Reward design  (v2 — no action-space cap)
-------------------------------------------
The reward alone produces creeping behavior without any external
speed cap in the driver.  The key mechanism is speed-conditioned
progress: the agent only earns progress reward when moving at the
appropriate speed for its distance to J1.

Target behaviors (in order):
  1. Approach at moderate speed, decelerate near J1
  2. Enter creep zone at low speed (1-3 m/s)
  3. Maintain low speed assertion -- signals intent to crossing drivers
  4. If a crossing driver brakes (yield detected) -- commit into the gap
  5. If no yield -- wait for a natural gap, then commit
  6. Complete crossing quickly once committed

Components:
  r_step      = -0.01                      tiny step cost
  r_safety    = f(TTC)                     0 when safe, -8 when ttc<0.5s
  r_approach  = -2.5*(excess/vmax)^2*prox  quadratic approach penalty
  r_creep     = bell(speed) - overspeed    Gaussian at 2 m/s + penalty above 4 m/s
  r_yield     = +2.0 max                   when crossing vehicle brakes while ego creeps
  r_progress  = 2*prog*speed_match         conditioned on speed appropriateness
  r_gap       = 0.3*gap_quality            small gap positioning signal
  terminal: +15 goal, -15 collision, -5 timeout
"""

import os
import sys
import math
import numpy as np
from typing import Tuple, List, Optional, Set

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

from .settings import MAX_ACCEL, MAX_DECEL, MAX_SPEED

Vec2 = Tuple[float, float]

# ─────────────────────────────────────────────────────────────────
# Lane constants  (Scenario 3, ego: -E2 → -E0, left turn)
# ─────────────────────────────────────────────────────────────────
EGO_APPROACH_LANE    = "-E2_0"
EGO_EXIT_LANE        = "-E0_0"       # ego exits left onto -E0
EGO_INTERNAL_LANE    = ":J1_3_0"    # internal lane for -E2 → -E0

CROSSING_APPROACH_LANES = {"E0_0", "-E1_0"}
CROSSING_INTERNAL_LANES = {":J1_5_0", ":J1_0_0", ":J1_1_0", ":J1_6_0"}
CROSSING_EXIT_LANES     = {"E1_0", "-E0_0"}

_DEFAULT_APPROACH_LEN = 59.56   # updated if SUMO reports differently
VEHICLE_CLEAR_TIME    = 1.5     # s for a vehicle to clear the conflict zone
EXPECTED_HEADWAY      = 5.14    # s at 700 veh/hr

OBS_DIM = 18

# Approach zone boundaries
APPROACH_ZONE_FAR  = 30.0  # m: start penalising excess speed from here
APPROACH_ZONE_NEAR = 12.0  # m: creep zone starts here (= creep_zone_dist)
TARGET_SPEED_AT_CREEP_ENTRY = 4.0   # m/s: desired speed at creep zone entry
OPTIMAL_CREEP_SPEED         = 2.0   # m/s: peak of creep reward bell curve
CREEP_BELL_WIDTH            = 1.5   # m/s: std of bell curve


# ─────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────
def _calculate_ttc(
    ego_pos: Vec2, ego_vel: Vec2,
    others_pos: List[Vec2], others_vel: List[Vec2],
    collision_distance: float = 2.5,
    max_ttc: float = 10.0,
    eps: float = 1e-9,
) -> float:
    ex, ey = ego_pos
    evx, evy = ego_vel
    d2 = collision_distance ** 2
    best = max_ttc

    for (ovx, ovy), (ox, oy) in zip(others_vel, others_pos):
        rx, ry = ox - ex, oy - ey
        if rx*rx + ry*ry <= d2:
            return 0.0
        vx, vy = ovx - evx, ovy - evy
        a = vx*vx + vy*vy
        if a <= eps:
            continue
        b = 2.0*(rx*vx + ry*vy)
        c = rx*rx + ry*ry - d2
        disc = b*b - 4.0*a*c
        if disc < 0.0:
            continue
        sq = math.sqrt(disc)
        cands = [t for t in ((-b-sq)/(2*a), (-b+sq)/(2*a)) if t >= 0.0]
        if cands:
            best = min(best, min(cands))
    return best


def _get_vehicle_state(veh_id: str) -> Tuple[Vec2, Vec2]:
    x, y = traci.vehicle.getPosition(veh_id)
    try:
        vx, vy = traci.vehicle.getVelocity(veh_id)
    except Exception:
        spd = traci.vehicle.getSpeed(veh_id)
        ang = math.radians(traci.vehicle.getAngle(veh_id))
        vx, vy = spd * math.sin(ang), spd * math.cos(ang)
    return (float(x), float(y)), (float(vx), float(vy))


def _target_speed_for_distance(dist_j1: float, creep_zone_dist: float) -> float:
    """
    Returns the appropriate target speed given distance to junction.
    Used both for the approach reward and obs[16].
    """
    if not math.isfinite(dist_j1) or dist_j1 > APPROACH_ZONE_FAR:
        return MAX_SPEED  # no constraint far away
    if dist_j1 <= creep_zone_dist:
        return OPTIMAL_CREEP_SPEED  # 2 m/s in the creep zone
    # Linear ramp from MAX_SPEED at APPROACH_ZONE_FAR to TARGET_SPEED_AT_CREEP_ENTRY at creep zone
    frac = (dist_j1 - creep_zone_dist) / (APPROACH_ZONE_FAR - creep_zone_dist)
    return TARGET_SPEED_AT_CREEP_ENTRY + frac * (MAX_SPEED - TARGET_SPEED_AT_CREEP_ENTRY)


# ─────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────
class SumoEnv:
    def __init__(
        self,
        cfg_path: str,
        use_gui: bool = False,
        step_length: float = 0.05,
        delay_ms: int = 0,
        lateral_resolution: float = 0.02,
        ego_id: str = "ego",
        nearby_radius: float = 80.0,
        collision_distance: float = 2.5,
        max_ttc: float = 10.0,
        ego_route_edges: Optional[List[str]] = None,
        ego_depart_lane: str = "0",
        ego_depart_pos: str = "0",
        ego_depart_speed: str = "0",
        ego_type_id: str = "DEFAULT_VEHTYPE",
        ego_speed_mode: int = 0,
        ego_lane_change_mode: int = 0,
        entry_junction_ids: Optional[List[str]] = None,
        creep_zone_dist: float = 12.0,
        ttc_critical: float = 2.0,
        timeout_steps_in_zone: int = 600,
        aggression_alpha: float = 1.0,   # unused, kept for API compat
    ):
        self.cfg_path           = cfg_path
        self.use_gui            = use_gui
        self.step_length        = step_length
        self.delay_ms           = delay_ms
        self.lateral_resolution = lateral_resolution
        self.ego_id             = ego_id
        self.nearby_radius      = nearby_radius
        self.collision_distance = collision_distance
        self.max_ttc            = max_ttc
        self.ego_route_edges    = ego_route_edges or ["-E2", "-E0"]
        self.ego_depart_lane    = ego_depart_lane
        self.ego_depart_pos     = ego_depart_pos
        self.ego_depart_speed   = ego_depart_speed
        self.ego_type_id        = ego_type_id
        self.ego_speed_mode     = ego_speed_mode
        self.ego_lane_change_mode = ego_lane_change_mode
        self.entry_junction_ids: Set[str] = set(entry_junction_ids or ["J1"])
        self.creep_zone_dist    = float(creep_zone_dist)
        self.ttc_critical       = float(ttc_critical)
        self.timeout_steps      = int(timeout_steps_in_zone)

        self.obs_dim = OBS_DIM

        self._started               = False
        self._ego_spawned_once      = False
        self._approach_length       = _DEFAULT_APPROACH_LEN
        self._prev_dist_to_route_end = None
        self._steps_in_creep_zone   = 0
        self._last_accel_cmd        = 0.0
        self._prev_ttc              = max_ttc

    # ─────────────────────────────────────────────
    # SUMO lifecycle
    # ─────────────────────────────────────────────
    def _build_cmd(self) -> List[str]:
        binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length",        str(self.step_length),
            "--lateral-resolution", str(self.lateral_resolution),
            "--end",                "10000",
            "--start",
            "--collision.action",   "warn",
            "--no-warnings",        "true",
        ]
        if self.use_gui and self.delay_ms > 0:
            cmd += ["--delay", str(self.delay_ms)]
        return cmd

    def start(self) -> None:
        if self._started:
            return
        traci.start(self._build_cmd(), verbose=False)
        self._started = True
        self._read_approach_length()

    def close(self) -> None:
        if self._started:
            try:
                traci.close(False)
            except Exception:
                pass
        self._started = False

    def _read_approach_length(self) -> None:
        try:
            self._approach_length = float(traci.lane.getLength(EGO_APPROACH_LANE))
        except Exception:
            self._approach_length = _DEFAULT_APPROACH_LEN

    # ─────────────────────────────────────────────
    # Ego management
    # ─────────────────────────────────────────────
    def _ego_exists(self) -> bool:
        return self.ego_id in traci.vehicle.getIDList()

    def _ensure_ego_modes(self) -> None:
        if not self._ego_exists():
            return
        traci.vehicle.setSpeedMode(self.ego_id, self.ego_speed_mode)
        traci.vehicle.setLaneChangeMode(self.ego_id, self.ego_lane_change_mode)

    def _spawn_ego(self) -> None:
        if self._ego_exists():
            self._ego_spawned_once = True
            return
        route_id = "egoRoute"
        try:
            if route_id not in traci.route.getIDList():
                traci.route.add(route_id, self.ego_route_edges)
        except Exception:
            pass
        traci.vehicle.add(
            vehID=self.ego_id, routeID=route_id, typeID=self.ego_type_id,
            depart="now", departLane=self.ego_depart_lane,
            departPos=self.ego_depart_pos, departSpeed=self.ego_depart_speed,
        )
        self._ego_spawned_once = True
        for _ in range(20):
            traci.simulationStep()
            if self._ego_exists():
                break

    # ─────────────────────────────────────────────
    # Distance helpers
    # ─────────────────────────────────────────────
    def _ego_lane(self) -> str:
        try:
            return str(traci.vehicle.getLaneID(self.ego_id))
        except Exception:
            return ""

    def _ego_dist_to_j1(self) -> float:
        if not self._ego_exists():
            return float("inf")
        lane = self._ego_lane()
        if lane == EGO_APPROACH_LANE:
            pos    = float(traci.vehicle.getLanePosition(self.ego_id))
            length = float(traci.lane.getLength(EGO_APPROACH_LANE))
            return max(0.0, length - pos)
        if any(f":{jid}" in lane for jid in self.entry_junction_ids):
            return 0.0
        return float("inf")

    def _dist_to_route_end(self) -> float:
        if not self._ego_exists():
            return 0.0
        try:
            lane_id  = traci.vehicle.getLaneID(self.ego_id)
            lane_pos = float(traci.vehicle.getLanePosition(self.ego_id))
            route    = traci.vehicle.getRoute(self.ego_id)
            idx      = traci.vehicle.getRouteIndex(self.ego_id)
            if not route or idx < 0:
                return 0.0
            curr_len = float(traci.lane.getLength(lane_id))
            rem = max(0.0, curr_len - lane_pos)
            for e in route[idx+1:]:
                try:
                    rem += float(traci.lane.getLength(f"{e}_0"))
                except Exception:
                    pass
            return float(rem)
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────
    # Crossing vehicle sensing
    # ─────────────────────────────────────────────
    def _get_crossing_vehicles(self) -> List[Tuple[float, float, float]]:
        """
        Returns (tta_s, speed_ms, accel_ms2) for each crossing vehicle,
        sorted by time-to-arrive at J1. Excludes vehicles already past J1.
        """
        vehs = []
        for vid in traci.vehicle.getIDList():
            if vid == self.ego_id:
                continue
            try:
                lane = str(traci.vehicle.getLaneID(vid))
            except Exception:
                continue

            if lane in CROSSING_EXIT_LANES:
                continue
            if lane in {EGO_APPROACH_LANE, EGO_INTERNAL_LANE}:
                continue

            is_approach = lane in CROSSING_APPROACH_LANES
            is_internal = lane in CROSSING_INTERNAL_LANES
            if not is_approach and not is_internal:
                continue

            try:
                speed = float(traci.vehicle.getSpeed(vid))
                try:
                    accel = float(traci.vehicle.getAcceleration(vid))
                except Exception:
                    accel = 0.0
            except Exception:
                continue

            if is_internal:
                tta = 0.0
            else:
                try:
                    lane_pos = float(traci.vehicle.getLanePosition(vid))
                    lane_len = float(traci.lane.getLength(lane))
                    dist = max(0.0, lane_len - lane_pos)
                    tta = dist / speed if speed > 0.1 else 99.0
                except Exception:
                    continue

            if math.isfinite(tta):
                vehs.append((tta, speed, accel))

        vehs.sort(key=lambda x: x[0])
        return vehs

    def _compute_gaps(
        self, crossing_vehs: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Returns (gap_size_s, gap_tta_s) for the 2 best upcoming gaps.
        gap_size_s  = duration of the gap window
        gap_tta_s   = time until the gap opens
        """
        if not crossing_vehs:
            return [(EXPECTED_HEADWAY * 2, 0.0), (EXPECTED_HEADWAY * 2, 0.0)]

        if len(crossing_vehs) == 1:
            tta, _, _ = crossing_vehs[0]
            gaps = []
            if tta > 0.5:
                gaps.append((tta, 0.0))
            gaps.append((EXPECTED_HEADWAY, tta + VEHICLE_CLEAR_TIME))
            gaps.sort(key=lambda x: x[1])
            return gaps[:2]

        gaps = []
        for i in range(len(crossing_vehs) - 1):
            tta_front = crossing_vehs[i][0]
            tta_rear  = crossing_vehs[i+1][0]
            gap_size  = max(0.0, tta_rear - tta_front - VEHICLE_CLEAR_TIME)
            gap_tta   = tta_front + VEHICLE_CLEAR_TIME
            gaps.append((gap_size, gap_tta))

        tta_last = crossing_vehs[-1][0]
        gaps.append((EXPECTED_HEADWAY, tta_last + VEHICLE_CLEAR_TIME))

        tta_first = crossing_vehs[0][0]
        if tta_first > 0.5:
            gaps.append((tta_first, 0.0))

        gaps.sort(key=lambda x: x[1])
        return gaps[:2]

    # ─────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────
    def _get_obs(self) -> dict:
        if not self._ego_exists():
            return {
                "ego_pos": (0., 0.), "ego_vel": (0., 0.),
                "ttc": self.max_ttc, "n_others": 0,
                "state": np.zeros(OBS_DIM, dtype=np.float32),
                "dist_j1": float("inf"), "in_creep": False,
                "cross_vehs": [], "gaps": [],
            }

        ego_pos, ego_vel = _get_vehicle_state(self.ego_id)
        ego_speed = math.sqrt(ego_vel[0]**2 + ego_vel[1]**2)
        dist_j1   = self._ego_dist_to_j1()
        in_creep  = math.isfinite(dist_j1) and dist_j1 <= self.creep_zone_dist

        # TTC vs ALL nearby vehicles
        nearby_pos, nearby_vel = [], []
        for vid in traci.vehicle.getIDList():
            if vid == self.ego_id:
                continue
            try:
                pos, vel = _get_vehicle_state(vid)
                dx = pos[0] - ego_pos[0]
                dy = pos[1] - ego_pos[1]
                if dx*dx + dy*dy <= self.nearby_radius**2:
                    nearby_pos.append(pos)
                    nearby_vel.append(vel)
            except Exception:
                continue

        ttc = _calculate_ttc(
            ego_pos, ego_vel, nearby_pos, nearby_vel,
            collision_distance=self.collision_distance,
            max_ttc=self.max_ttc,
        )

        cross_vehs = self._get_crossing_vehicles()
        gaps       = self._compute_gaps(cross_vehs)

        MAX_TTA = 10.0
        MAX_GAP = 10.0

        def _n_tta(t): return float(np.clip(t/MAX_TTA, 0., 1.)) if math.isfinite(t) else 1.0
        def _n_spd(s): return float(np.clip(s/MAX_SPEED, 0., 1.))
        def _n_acc(a): return float(np.clip(a/4.0, -1., 1.))

        # [0-3] ego state
        d_norm = float(np.clip(dist_j1/self._approach_length, 0., 1.)) if math.isfinite(dist_j1) else 1.0
        obs = [d_norm, _n_spd(ego_speed), float(np.clip(ttc/self.max_ttc, 0., 1.)), float(in_creep)]

        # [4-11] 2 nearest crossing vehicles
        for i in range(2):
            if i < len(cross_vehs):
                tta, spd, acc = cross_vehs[i]
                obs += [1.0, _n_tta(tta), _n_spd(spd), _n_acc(acc)]
            else:
                obs += [0.0, 1.0, 0.0, 0.0]

        # [12-15] 2 gap features
        for i in range(2):
            if i < len(gaps):
                gsize, gtta = gaps[i]
                obs += [
                    float(np.clip(gsize/MAX_GAP, 0., 1.)) if math.isfinite(gsize) else 1.0,
                    float(np.clip(gtta/MAX_TTA,  0., 1.)) if math.isfinite(gtta)  else 1.0,
                ]
            else:
                obs += [1.0, 1.0]

        # [16] approach_speed_excess_norm
        # Positive → going too fast for current distance to J1
        # This is the key signal for the approach deceleration behavior
        target_spd = _target_speed_for_distance(dist_j1, self.creep_zone_dist)
        speed_excess = (ego_speed - target_spd) / MAX_SPEED
        obs.append(float(np.clip(speed_excess, -1., 1.)))

        # [17] time_in_zone_norm (urgency/timeout signal)
        obs.append(float(np.clip(self._steps_in_creep_zone / self.timeout_steps, 0., 1.)))

        assert len(obs) == OBS_DIM, f"obs dim mismatch: {len(obs)} vs {OBS_DIM}"

        return {
            "ego_pos":    ego_pos,
            "ego_vel":    ego_vel,
            "ttc":        float(ttc),
            "n_others":   len(nearby_pos),
            "state":      np.array(obs, dtype=np.float32),
            "dist_j1":    dist_j1,
            "in_creep":   in_creep,
            "cross_vehs": cross_vehs,
            "gaps":       gaps,
        }

    # ─────────────────────────────────────────────
    # Reset / Step
    # ─────────────────────────────────────────────
    def reset(self) -> dict:
        if not self._started:
            self.start()
        else:
            traci.load([
                "-c", self.cfg_path,
                "--step-length",        str(self.step_length),
                "--lateral-resolution", str(self.lateral_resolution),
                "--start",
                "--no-warnings", "true",
            ])

        self._ego_spawned_once       = False
        self._prev_dist_to_route_end = None
        self._steps_in_creep_zone    = 0
        self._last_accel_cmd         = 0.0
        self._prev_ttc               = self.max_ttc

        for _ in range(10):
            traci.simulationStep()

        self._spawn_ego()
        self._ensure_ego_modes()

        for _ in range(5):
            traci.simulationStep()

        return self._get_obs()

    def step(
        self, action: float, accel_cmd: float = 0.0
    ) -> Tuple[dict, float, bool, dict]:
        """
        action    : target speed (m/s), already zone-capped by driver
        accel_cmd : acceleration command (for bookkeeping)
        """
        self._last_accel_cmd = float(accel_cmd)

        if self._ego_exists():
            traci.vehicle.setSpeed(self.ego_id, max(0.0, float(action)))

        traci.simulationStep()

        # ── Collision ──────────────────────────────────────────────
        if self.ego_id in traci.simulation.getCollidingVehiclesIDList():
            obs = self._get_obs()
            info = {
                "ttc": obs["ttc"], "dist_j1": obs.get("dist_j1", 0.),
                "reached_goal": False, "collided": True, "timeout": False,
                "best_gap": 0., "r_safety": -15., "r_progress": 0.,
                "r_creep": 0., "r_approach": 0., "r_yield": 0., "reward": -15.,
            }
            return obs, -15.0, True, info

        obs      = self._get_obs()
        ttc      = float(obs["ttc"])
        dist_j1  = obs.get("dist_j1", float("inf"))
        in_creep = obs.get("in_creep", False)
        gaps     = obs.get("gaps", [])
        cross_vehs = obs.get("cross_vehs", [])
        best_gap = float(gaps[0][0]) if gaps else 0.0

        ego_speed = 0.0
        if self._ego_exists():
            try:
                ego_speed = float(traci.vehicle.getSpeed(self.ego_id))
            except Exception:
                pass

        # ── Timeout ────────────────────────────────────────────────
        if in_creep:
            self._steps_in_creep_zone += 1
        else:
            self._steps_in_creep_zone = 0
        timeout = self._steps_in_creep_zone >= self.timeout_steps

        # ── Progress ───────────────────────────────────────────────
        dist_end = self._dist_to_route_end()
        if self._prev_dist_to_route_end is None or not math.isfinite(dist_end):
            progress = 0.0
        else:
            progress = max(0.0, self._prev_dist_to_route_end - dist_end)
        self._prev_dist_to_route_end = dist_end

        dt      = self.step_length
        v_max   = MAX_SPEED
        try:
            v_max = float(traci.vehicle.getAllowedSpeed(self.ego_id)) if self._ego_exists() else MAX_SPEED
        except Exception:
            pass
        max_prog  = max(1e-6, v_max * dt)
        prog_norm = float(np.clip(progress / max_prog, -1.0, 1.0))

        # ── Goal ───────────────────────────────────────────────────
        ego_on_exit = False
        if self._ego_exists():
            try:
                ego_on_exit = traci.vehicle.getLaneID(self.ego_id) == EGO_EXIT_LANE
            except Exception:
                pass
        reached_goal = ego_on_exit or (math.isfinite(dist_end) and dist_end < 2.0)

        # ════════════════════════════════════════════════════════════
        # REWARD  (v2 — no action-space cap, reward-only shaping)
        # ════════════════════════════════════════════════════════════
        #
        # Design rationale for cap removal:
        # The previous reward let the agent farm r_progress by going
        # fast near J1 because progress was unconditional on speed.
        # The weak linear r_approach (-0.5 * excess/MAX_SPEED) could
        # not overcome r_progress (2.0 * prog_norm ≈ 1.4 at 10 m/s).
        # The zone speed cap papered over this by mechanically limiting
        # speed.  Now we fix the root cause:
        #
        #   (A) SPEED-CONDITIONED PROGRESS — the agent only earns
        #       progress reward when moving at the appropriate speed
        #       for its distance to J1.  Going fast near the junction
        #       yields nearly zero progress reward.  This is the key
        #       change that makes the cap unnecessary.
        #
        #   (B) QUADRATIC APPROACH PENALTY — scales with proximity
        #       to junction, providing a dense gradient for braking.
        #
        #   (C) ASYMMETRIC CREEP REWARD — bell curve for positive
        #       incentive + explicit overspeed penalty for speeds
        #       above creep entry threshold.
        #
        # Numerical verification (dt=0.05, MAX_SPEED=13.89):
        #   At 12m from J1, target_spd=4 m/s:
        #     ego=4  → speed_match=1.0, r_progress≈+0.58, r_approach=0
        #     ego=8  → speed_match≈0.05, r_progress≈0.07, r_approach≈-0.27
        #     ego=12 → speed_match≈0.0,  r_progress≈0,    r_approach≈-0.72
        #   Agent cannot profit from rushing — reward landscape is
        #   self-enforcing without any external speed cap.

        # ── 1. Step cost ──────────────────────────────────────────
        r_step = -0.01

        # ── 2. Safety — sharp penalty for dangerous TTC ───────────
        if ttc < 0.5:
            r_safety = -8.0
        elif ttc < self.ttc_critical:
            r_safety = -8.0 * (self.ttc_critical - ttc) / (self.ttc_critical - 0.5)
        else:
            r_safety = 0.0

        # ── 3. Approach speed management (QUADRATIC + PROXIMITY) ──
        # Penalizes arriving at J1 too fast.  Quadratic in excess
        # speed so that small overshoots are tolerated but large
        # ones are heavily punished.  Weighted by proximity to J1
        # so the penalty intensifies as the agent gets closer.
        r_approach = 0.0
        if math.isfinite(dist_j1) and self.creep_zone_dist < dist_j1 < APPROACH_ZONE_FAR:
            target_spd = _target_speed_for_distance(dist_j1, self.creep_zone_dist)
            excess = max(0.0, ego_speed - target_spd)
            excess_norm = excess / MAX_SPEED
            # proximity: 0 at APPROACH_ZONE_FAR, 1 at creep zone entry
            proximity = 1.0 - (dist_j1 - self.creep_zone_dist) / (APPROACH_ZONE_FAR - self.creep_zone_dist)
            r_approach = -2.5 * excess_norm * excess_norm * (0.3 + 0.7 * proximity)

        # ── 4. Creep reward — ASYMMETRIC bell + overspeed penalty ─
        # Positive: Gaussian centered at 2 m/s rewards low-speed
        #   assertion (the negotiation signal to crossing drivers).
        # Negative: explicit penalty for entering the creep zone
        #   above the entry threshold.  Without this, the bell curve
        #   just gives zero at high speed — zero ≠ discouragement.
        #   The penalty makes high-speed entry actively harmful to
        #   the return, creating a clear gradient toward deceleration.
        r_creep = 0.0
        if in_creep and ego_speed > 0.3:
            if ttc >= 1.5:
                z = (ego_speed - OPTIMAL_CREEP_SPEED) / CREEP_BELL_WIDTH
                r_creep = 1.5 * math.exp(-0.5 * z * z)
            # Overspeed penalty in creep zone — active regardless of TTC
            if ego_speed > TARGET_SPEED_AT_CREEP_ENTRY:
                overspeed_norm = (ego_speed - TARGET_SPEED_AT_CREEP_ENTRY) / MAX_SPEED
                r_creep -= 2.0 * overspeed_norm * overspeed_norm

        # ── 5. Yield detection bonus ──────────────────────────────
        r_yield = 0.0
        if in_creep and cross_vehs and ego_speed > 0.3:
            nearest_accel = cross_vehs[0][2]  # (tta, speed, accel)
            if nearest_accel < -1.5:
                r_yield = min(2.0, abs(nearest_accel) / 1.5)

        # ── 6. Progress reward — SPEED-CONDITIONED ────────────────
        # This is the critical change that eliminates the need for
        # action-space speed caps.
        #
        # Insight: the old r_progress = 2*prog_norm rewarded raw
        # displacement regardless of speed.  At 10 m/s near J1 the
        # agent earned ~1.4/step — more than the approach penalty
        # could counteract.  The rational policy was to rush.
        #
        # Fix: multiply progress by a speed-appropriateness factor.
        # The factor is 1.0 when ego_speed matches the target speed
        # profile, and decays exponentially for overspeeding.
        # This makes it impossible to farm progress reward by going
        # fast near the junction — the reward landscape itself
        # enforces deceleration.
        #
        # speed_match function:
        #   ratio <= 1 (at or below target): linear ramp 0→1
        #   ratio >  1 (above target):       exp(-3*(ratio-1)^2)
        #     ratio=1.5 → match=0.47
        #     ratio=2.0 → match=0.05
        #     ratio=3.0 → match=0.0001
        #
        # In-junction (dist_j1=inf): target=MAX_SPEED, ratio≤1,
        # match=1.0 → full progress reward.  Agent commits freely.
        target_spd = _target_speed_for_distance(dist_j1, self.creep_zone_dist)
        if ego_speed <= 0.1:
            speed_match = 0.0
        elif target_spd >= MAX_SPEED - 0.1:
            # Far from junction or in junction — no speed constraint
            speed_match = 1.0
        else:
            speed_ratio = ego_speed / max(target_spd, 0.5)
            if speed_ratio <= 1.0:
                speed_match = speed_ratio
            else:
                speed_match = math.exp(-3.0 * (speed_ratio - 1.0) ** 2)

        if ttc >= self.ttc_critical:
            r_progress = 2.0 * prog_norm * speed_match
        elif ttc >= 1.0:
            r_progress = 0.0
        else:
            # Dangerous TTC: penalize fast forward motion
            r_progress = -1.0 * prog_norm

        # ── 7. Gap quality ────────────────────────────────────────
        r_gap = 0.0
        if in_creep and math.isfinite(best_gap) and best_gap > 0:
            r_gap = 0.3 * float(np.clip(best_gap / 5.0, 0.0, 1.0))

        # ── Total ─────────────────────────────────────────────────
        reward = r_step + r_safety + r_approach + r_creep + r_yield + r_progress + r_gap

        # Terminal bonuses
        if reached_goal:
            reward += 15.0
        if timeout:
            reward -= 5.0

        self._prev_ttc = ttc
        ego_gone = self._ego_spawned_once and not self._ego_exists()
        done = ego_gone or reached_goal or timeout

        info = {
            "ttc":          ttc,
            "dist_j1":      dist_j1,
            "reached_goal": reached_goal,
            "collided":     False,
            "timeout":      timeout,
            "best_gap":     best_gap,
            "r_safety":     r_safety,
            "r_approach":   r_approach,
            "r_creep":      r_creep,
            "r_yield":      r_yield,
            "r_progress":   r_progress,
            "r_gap":        r_gap,
            "speed_match":  speed_match,
            "reward":       reward,
        }
        return obs, float(reward), bool(done), info