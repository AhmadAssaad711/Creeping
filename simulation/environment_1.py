"""
Minimal SUMO/TraCI script for computing TTC each step (for reward shaping).

EDIT (as you asked):
- Ego spawns on the SAME explicit edges route: ["-E2", "-E0"]
- Observation now includes a fixed-size RL-ready state vector:
  ego (pos, vel, speed), TTC, and K=1 closest agent relative features (dx,dy,dvx,dvy,dist,closing_speed)
"""

import os
import sys
import math
import argparse
import numpy as np
from typing import Tuple, List, Optional


# -------------------------
# SUMO / TraCI setup
# -------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

Vec2 = Tuple[float, float]


# -------------------------
# TTC computation
# -------------------------
def calculate_ttc(
    ego_speed: Vec2,
    ego_pos: Vec2,
    others_speed: List[Vec2],
    others_pos: List[Vec2],
    collision_distance: float = 2.5,  # meters (tune)
    max_ttc: float = 20.0,            # seconds (cap)
    eps: float = 1e-9,
) -> float:
    ex, ey = ego_pos
    evx, evy = ego_speed

    d2 = collision_distance * collision_distance
    best = max_ttc

    for (ovx, ovy), (ox, oy) in zip(others_speed, others_pos):
        rx = ox - ex
        ry = oy - ey

        # already too close
        if (rx * rx + ry * ry) <= d2:
            return 0.0

        vx = ovx - evx
        vy = ovy - evy

        # a t^2 + b t + c = 0
        a = vx * vx + vy * vy
        b = 2.0 * (rx * vx + ry * vy)
        c = (rx * rx + ry * ry) - d2

        if a <= eps:
            continue

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        candidates = [t for t in (t1, t2) if t >= 0.0]
        if not candidates:
            continue

        ttc = min(candidates)
        if ttc < best:
            best = ttc

    return best


def _speed_vec_from_speed_angle(speed: float, angle_deg: float) -> Vec2:
    th = math.radians(angle_deg)
    vx = speed * math.cos(th)
    vy = speed * math.sin(th)
    return vx, vy


def get_vehicle_state(veh_id: str) -> Tuple[Vec2, Vec2]:
    x, y = traci.vehicle.getPosition(veh_id)

    try:
        vx, vy = traci.vehicle.getVelocity(veh_id)
    except Exception:
        speed = traci.vehicle.getSpeed(veh_id)
        angle = traci.vehicle.getAngle(veh_id)
        vx, vy = _speed_vec_from_speed_angle(speed, angle)

    return (float(x), float(y)), (float(vx), float(vy))


# -------------------------
# Environment wrapper
# -------------------------
class SumoEnv:
    def __init__(
        self,
        cfg_path: str,
        use_gui: bool = True,
        step_length: float = 0.05,
        delay_ms: int = 0,
        lateral_resolution: float = 0.1,
        ego_id: str = "ego",
        nearby_radius: float = 80.0,
        collision_distance: float = 2.5,
        max_ttc: float = 20.0,
        # --- minimal additions for ego init/route ---
        ego_from_edge: Optional[str] = None,
        ego_to_edge: Optional[str] = None,
        ego_depart_lane: str = "0",
        ego_depart_pos: str = "0",
        ego_depart_speed: str = "0",
        ego_type_id: str = "DEFAULT_VEHTYPE",
        # --- explicit route edges ---
        ego_route_edges: List[str] = None,
        # --- control modes ---
        ego_speed_mode: int = 0,
        ego_lane_change_mode: int = 0,
        # --- RL observation config (NEW) ---
        K_closest: int = 1,  # you asked: closest agent is what we care about
    ):
        self.cfg_path = cfg_path
        self.use_gui = use_gui
        self.step_length = step_length
        self.delay_ms = delay_ms
        self.lateral_resolution = lateral_resolution

        self.ego_id = ego_id
        self.nearby_radius = nearby_radius
        self.collision_distance = collision_distance
        self.max_ttc = max_ttc

        self.ego_from_edge = ego_from_edge
        self.ego_to_edge = ego_to_edge
        self.ego_depart_lane = ego_depart_lane
        self.ego_depart_pos = ego_depart_pos
        self.ego_depart_speed = ego_depart_speed
        self.ego_type_id = ego_type_id

        self.ego_route_edges = ego_route_edges

        self.ego_speed_mode = ego_speed_mode
        self.ego_lane_change_mode = ego_lane_change_mode

        self.K_closest = int(K_closest)

        self._started = False
        self._ego_has_spawned_once = False

        self.prev_dist_to_goal= None

        self.goal_edge = "-E0"
        self.goal_eps = 2.0          # meters from the end to count as success


    def _build_sumo_cmd(self) -> List[str]:
        sumoBinary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        cmd = [
            sumoBinary,
            "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            "--lateral-resolution", str(self.lateral_resolution),
            "--end", "10000",
            "--start",
        ]
        if self.use_gui and self.delay_ms > 0:
            cmd += ["--delay", str(self.delay_ms)]
        return cmd

    def start(self) -> None:
        if self._started:
            return
        traci.start(self._build_sumo_cmd(), verbose=True)
        self._started = True

    def close(self) -> None:
        if self._started:
            try:
                traci.close(False)
            except Exception:
                pass
        self._started = False

    def _ego_exists(self) -> bool:
        return self.ego_id in traci.vehicle.getIDList()

    def _ensure_ego_control_modes(self) -> None:
        if not self._ego_exists():
            return
        traci.vehicle.setSpeedMode(self.ego_id, int(self.ego_speed_mode))
        traci.vehicle.setLaneChangeMode(self.ego_id, int(self.ego_lane_change_mode))

    def _spawn_ego_if_missing(self) -> None:
        if self._ego_exists():
            self._ego_has_spawned_once = True
            return

        edges = self.ego_route_edges if self.ego_route_edges else None

        if not edges:
            if not self.ego_from_edge or not self.ego_to_edge:
                return
            route_res = traci.simulation.findRoute(self.ego_from_edge, self.ego_to_edge)
            edges = list(route_res.edges) if route_res is not None else []
            if not edges:
                raise RuntimeError(
                    f"findRoute failed: fromEdge='{self.ego_from_edge}' toEdge='{self.ego_to_edge}'. "
                    f"Check your edge IDs in the network."
                )

        route_id = "egoRoute"
        try:
            traci.route.add(route_id, edges)
        except Exception:
            pass

        traci.vehicle.add(
            vehID=self.ego_id,
            routeID=route_id,
            typeID=self.ego_type_id,
            depart="now",
            departLane=self.ego_depart_lane,
            departPos=self.ego_depart_pos,
            departSpeed=self.ego_depart_speed,
        )

        self._ego_has_spawned_once = True

        for _ in range(20):
            traci.simulationStep()
            if self._ego_exists():
                break

    def _dist_to_route_end(self) -> float:
        """Remaining distance (meters) along the ego's current route to its final edge."""
        if not self._ego_exists():
            return 0.0

        # current lane position on current edge
        lane_id = traci.vehicle.getLaneID(self.ego_id)
        lane_pos = traci.vehicle.getLanePosition(self.ego_id)

        # current edge id (lane is like 'edge_0', edge is 'edge')
        edge_id = traci.vehicle.getRoadID(self.ego_id)

        route = traci.vehicle.getRoute(self.ego_id)          # list of edge IDs
        idx = traci.vehicle.getRouteIndex(self.ego_id)       # current edge index in route

        if route is None or len(route) == 0 or idx < 0:
            return 0.0

        # remaining on current edge
        # NOTE: lane length may differ from edge length, but it's usually fine.
        curr_lane_len = traci.lane.getLength(lane_id)
        rem = max(0.0, float(curr_lane_len - lane_pos))

        # add full lengths of remaining future edges
        for e in route[idx + 1:]:
            try:
                rem += self._edge_length(e)
            except Exception:
                pass

        return float(rem)

    def _edge_length(self, edge_id: str) -> float:
        # simplest: use lane 0 length as edge length
        lane0 = f"{edge_id}_0"
        return float(traci.lane.getLength(lane0))

    def _dist_to_end_of_edge(self, goal_edge: str) -> float:
        """Remaining distance (m) until the end of goal_edge along the ego's current route."""
        if not self._ego_exists():
            return 0.0

        route = traci.vehicle.getRoute(self.ego_id)          # list of edge ids
        idx   = traci.vehicle.getRouteIndex(self.ego_id)     # current edge index
        if not route or idx < 0:
            return 0.0

        if goal_edge not in route:
            # if route doesn't contain goal edge, you can treat as "far"
            return float("inf")

        goal_i = route.index(goal_edge)

        # If we've already passed the goal edge in the route, remaining is 0
        if idx > goal_i:
            return 0.0

        # Remaining on current lane
        lane_id  = traci.vehicle.getLaneID(self.ego_id)
        lane_pos = traci.vehicle.getLanePosition(self.ego_id)
        rem = max(0.0, float(traci.lane.getLength(lane_id) - lane_pos))

        # Add full lengths of intermediate edges up to and including the goal edge
        for e in route[idx + 1 : goal_i + 1]:
            rem += self._edge_length(e)

        return float(rem)

    def reset(self) -> dict:
        if not self._started:
            self.start()
        else:
            traci.load([
                "-c", self.cfg_path,
                "--step-length", str(self.step_length),
                "--lateral-resolution", str(self.lateral_resolution),
                "--start",
            ])

        self._ego_has_spawned_once = False

        self.prev_dist_to_goal= None


        # Step a bit so GUI visibly advances without pressing Play
        for _ in range(10):
            traci.simulationStep()

        self._spawn_ego_if_missing()
        self._ensure_ego_control_modes()

        # Optional: step again so ego is definitely inserted & moving
        for _ in range(5):
            traci.simulationStep()

        return self._get_obs()



    def step(self, action: float) -> Tuple[dict, float, bool, dict]:
        if self._ego_exists():
            target_speed = max(0.0, float(action))
            traci.vehicle.setSpeed(self.ego_id, target_speed)

        traci.simulationStep()

        # ---- Collision check (end episode on collision) ----
        coll_ids = traci.simulation.getCollidingVehiclesIDList()
        collided = (self.ego_id in coll_ids)
        if collided:
            obs = self._get_obs()
            reward = -100.0
            done = True
            info = {"ttc": obs.get("ttc", 0.0), "collided": True, "dist_to_goal": None, "reached_goal": False}
            return obs, float(reward), bool(done), info

        # compute obs + TTC
        obs = self._get_obs()
        ttc = float(obs["ttc"])

        # --- TTC penalty ---
        if ttc <= 2.0:
            r_ttc = -1.0 * (2.0 - ttc)
        else:
            r_ttc = -1.0 / (ttc + 1e-6)

        # --- Distance to route end ---
        dist_to_goal = self._dist_to_end_of_edge(self.goal_edge)

        # progress toward the end of -E0
        if self.prev_dist_to_goal is None or not np.isfinite(dist_to_goal):
            progress = 0.0
        else:
            progress = self.prev_dist_to_goal - dist_to_goal

        self.prev_dist_to_goal = dist_to_goal
        r_progress = float(progress)

        reached_goal = np.isfinite(dist_to_goal) and (dist_to_goal < self.goal_eps)

        reward = r_ttc + 10.0 * r_progress
        if reached_goal:
            reward += 50.0   # success bonus (tune)

        ego_gone = self._ego_has_spawned_once and (not self._ego_exists())

        done = ego_gone or reached_goal
        info = {"ttc": ttc, "dist_to_goal": dist_to_goal, "reached_goal": reached_goal, "collided": collided}
        return obs, float(reward), bool(done), info

    # -------------------------
    # RL-ready observation (FIXED)
    # -------------------------
    def _get_obs(self) -> dict:
        ids = traci.vehicle.getIDList()
        if self.ego_id not in ids:
            # Keep dict structure, but also provide fixed-size state vector.
            state = np.zeros((3 + 2 + 1 + 1 + self.K_closest * 7,), dtype=np.float32)
            # layout: [ego_x,ego_y,ego_speed, ego_vx,ego_vy, ttc, n_others, K*(exists,dx,dy,dvx,dvy,dist,closing)]
            return {
                "ego_pos": (0.0, 0.0),
                "ego_vel": (0.0, 0.0),
                "ttc": self.max_ttc,
                "n_others": 0,
                "state": state,
            }

        ego_pos, ego_vel = get_vehicle_state(self.ego_id)
        ex, ey = ego_pos
        evx, evy = ego_vel
        ego_speed = float(math.sqrt(evx * evx + evy * evy))

        others = [vid for vid in ids if vid != self.ego_id]

        # collect nearby vehicles (within radius)
        r2_max = self.nearby_radius * self.nearby_radius
        near: List[Tuple[float, str, Vec2, Vec2]] = []  # (dist2, id, pos, vel)

        for vid in others:
            pos, vel = get_vehicle_state(vid)
            dx = pos[0] - ex
            dy = pos[1] - ey
            d2 = dx * dx + dy * dy
            if d2 <= r2_max:
                near.append((d2, vid, pos, vel))

        # Sort by closest distance (your request)
        near.sort(key=lambda t: t[0])

        # TTC uses all nearby vehicles (same as before)
        near_pos = [t[2] for t in near]
        near_vel = [t[3] for t in near]

        ttc = calculate_ttc(
            ego_speed=ego_vel,
            ego_pos=ego_pos,
            others_speed=near_vel,
            others_pos=near_pos,
            collision_distance=self.collision_distance,
            max_ttc=self.max_ttc,
        )

        # Build fixed-size RL vector
        # Base features
        n_others = len(near)
        base = [
            float(ex), float(ey), float(ego_speed),
            float(evx), float(evy),
            float(ttc),
            float(n_others),
        ]

        # Closest-K features (each: exists, dx, dy, dvx, dvy, dist, closing_speed)
        # closing_speed > 0 means "closing in"
        closest_feats: List[float] = []
        dt = float(self.step_length)
        eps = 1e-9

        for i in range(self.K_closest):
            if i < len(near):
                _, _, (ox, oy), (ovx, ovy) = near[i]
                dx = float(ox - ex)
                dy = float(oy - ey)
                dvx = float(ovx - evx)
                dvy = float(ovy - evy)
                dist = float(math.sqrt(dx * dx + dy * dy) + eps)
                closing = float(-(dx * dvx + dy * dvy) / dist)  # projection
                closest_feats += [1.0, dx, dy, dvx, dvy, dist, closing]
            else:
                closest_feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        state = np.array(base + closest_feats, dtype=np.float32)

        return {
            "ego_pos": ego_pos,
            "ego_vel": ego_vel,
            "ttc": float(ttc),
            "n_others": n_others,
            "state": state,
        }