# ----------------------------- CARLA PythonAPI on sys.path -----------------------------
import os, sys, glob, math, random, time
import numpy as np

def _default_carla_root():
    if sys.platform.startswith("win"):
        return r"C:/Users/hp/Desktop/Carla/CARLA_0.9.14/WindowsNoEditor"
    elif sys.platform.startswith("linux"):
        # CHANGE THIS to where your CARLA folder is on Ubuntu
        # examples: ~/CARLA_0.9.14  OR  ~/carla/CARLA_0.9.14
        return os.path.expanduser("~/CARLA_0.9.14")
    else:
        return ""

CARLA_ROOT = os.environ.get("CARLA_ROOT", _default_carla_root())

# If carla is installed via pip and import works, you can skip all of this.
# But if you're using CARLA's PythonAPI from a local CARLA install, keep it.
egg_globs = []
if sys.platform.startswith("win"):
    egg_globs = [
        os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-*-py3.*-win-amd64.egg"),
        os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-*cp3*-win-amd64.egg"),
    ]
elif sys.platform.startswith("linux"):
    egg_globs = [
        os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-*-py3.*-linux-x86_64.egg"),
        os.path.join(CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-*cp3*-linux-x86_64.egg"),
    ]

eggs = []
for pattern in egg_globs:
    eggs.extend(glob.glob(pattern))

if eggs:
    egg_path = sorted(eggs)[-1]
    if egg_path not in sys.path:
        sys.path.insert(0, egg_path)

pythonapi_dir = os.path.join(CARLA_ROOT, "PythonAPI")
carla_pkg_dir = os.path.join(pythonapi_dir, "carla")
for p in (pythonapi_dir, carla_pkg_dir):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController


# Safe default if not provided in settings.py
try:
    CAR_NAME  # type: ignore[name-defined]
except NameError:
    CAR_NAME = os.environ.get("CARLA_EGO_BPNAME", "vehicle.tesla.model3")

from .sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from .settings import *  # VISUAL_DISPLAY, CAR_NAME, MAX_ACCEL, MAX_DECEL, W_* knobs, etc.

class CarlaEnvironment:
    def __init__(self, client, world, town, episode_length):
        """
    Phase-1: cautious gap negotiation at an unsignalized conflict zone (single crossing car, no occluders).

    Ego control:
      - RL controls longitudinal acceleration a_cmd in [-MAX_DECEL, +MAX_ACCEL] (m/s^2) -> throttle/brake
      - CARLA PID controls steering to keep the ego on-lane

    Crossing car control:
      - Spawn at a fixed transform, then Traffic Manager (TM) autopilot
      - Random per-episode driving "style":
          * speed profile: conservative / normal / aggressive (via TM speed % difference)
          * interaction profile: respects ego (0% ignore) / partially ignores (~50%) / fully ignores (100%)
      - Always ignores traffic lights (and signs)
      - Forced to choose "straight" at branch points by setting a TM path made of "straightest" waypoints.

    Obs:
      returns [image_obs, nav_obs]
    """

    def __init__(self, client, world, town, episode_length):
        self.client = client
        self.world = world
        self.town = town

        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.display_on = VISUAL_DISPLAY

        # ---- sync + fixed dt ----
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            print("[Init] Enabling synchronous mode.")
            settings.synchronous_mode = True
        self.delta_seconds = 0.05
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)

        # ---- Traffic Manager ----
        self.tm_port = int(os.environ.get("TM_PORT", "8000"))
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        seed = int(os.environ.get("TM_SEED", "42"))
        self.tm.set_random_device_seed(seed)
        self.tm.set_global_distance_to_leading_vehicle(2.5)

        # ---- realtime pacing ----
        self.realtime = False

        # ---- action bounds ----
        self.max_accel = float(MAX_ACCEL)      # + m/s^2
        self.max_decel = float(MAX_DECEL)      # magnitude (positive); action min is -max_decel

        # ---- RL action interpretation: 1D "pedal" u in [-1, 1] ----
        # u > 0  => throttle (forward), u < 0 => brake, u ~ 0 => coast
        # We convert u -> physical accel proxy (m/s^2) for logging/reward terms.
        self.use_pedal_action = True

        # deadzone avoids tiny sign-flip noise turning into brake/throttle toggles
        self.pedal_deadzone = 0.05

        # slew-rate limit (max change in u per step); helps stop rapid +/-
        # With dt≈0.05, 0.08 means ~0.08/step => 1.6 per second.
        self.pedal_slew = 0.08

        # map pedal to actual CARLA controls
        self.throttle_gain = 1.5
        self.brake_gain = 1.0
        self.min_throttle = 0.20  # overcome static friction / inertia
        self.min_brake = 0.10

        # internal state for smoothing
        self._u_prev = 0.0

        # ---- reward knobs ----
        self.weight_of_safety = float(W_SAFETY)
        self.weight_of_negotiation_quality = float(W_NEGOTIATION)
        self.weight_of_efficiency = float(W_EFFICIENCY)
        self.weight_of_smoothness = float(W_SMOOTHNESS)

        # ---- episode state ----
        self.timesteps = 0
        self.prev_accel = 0.0
        self.ego_current_speed = 0.0
        self.distance_to_stop_line = 0.0
        self.prev_D = 0.0

        # ---- actors & sensors ----
        self.ego_vehicle = None
        #self.other_vehicles = None

        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None

        self.sensors_list = []
        self.actors_list = []

        self.image_obs = None
        self.navigation_obs = None
        self.collisions_history = []

        # ---- stop line segment (Town02 example – adjust as needed) ----
        stop_line_start = carla.Location(x=132.123, y=226.178, z=0.0)
        stop_line_end   = carla.Location(x=146.519, y=240.933, z=0.0)

        self.start_stop_line_wp = self.map.get_waypoint(
            stop_line_start, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        self.end_stop_line_wp = self.map.get_waypoint(
            stop_line_end, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # ---- PID steering (ego) ----
        self.pid_controller = None
        self.pid_args_lateral = {"K_P": 1.0, "K_D": 0.05, "K_I": 0.0, "dt": self.delta_seconds}
        self.pid_args_longitudinal = {"K_P": 1.0, "K_D": 0.0, "K_I": 0.0, "dt": self.delta_seconds}
        self.lookahead_dist = float(getattr(globals(), "LOOKAHEAD_DIST", 5.0))

        # ---- termination ----
        self.max_steps = episode_length
        self.success_clearance = float(getattr(globals(), "SUCCESS_CLEARANCE_METERS", 8.0))

        # route buffers
        self.start_wp = None
        self.end_wp = None
        self.route_waypoints = None
        self.current_waypoint_index = 0

        # Other agents
        # ---- multi-agent crossing vehicles (8) ----
        self.num_other_agents = int(getattr(globals(), "NUM_OTHER_AGENTS", 1))
        self.other_vehicles = [None] * self.num_other_agents

        # You said you know corresponding conflicts; keep placeholders and fill later.
        # Option A: store per-agent conflict point as (x,y)
        self.conflict_points_xy = [(132.981, 236.924)]  # two conflict points total; you fill later e.g. [(x1,y1),(x2,y2)]
        self.other_conflict_id = [0]  # each agent chooses 0 or 1, you fill mapping

        # Per-agent spawn transforms (you fill later)
        transform_2 = carla.Transform(
            carla.Location(164.521, 236.943, 0.000),
            carla.Rotation(pitch=0.00, yaw=-179.97, roll=0.00)
        )
        self.other_spawn_tfs = [transform_2]

        # Ego Priority
        self.ego_priority = 0

        # TTC
        self.ego_TTC = [float("inf")]
        self.other_TTC = [float("inf")] * self.num_other_agents
        # small gap → must wait
        self.ttc_critical = float(getattr(globals(), "TTC_CRITICAL", 1.5))

        # large gap → can go
        self.ttc_safe = float(getattr(globals(), "TTC_SAFE", 3.5))

        # bookkeeping
        self.entered_stopline = False
        self.crossed_intersection = False

        # --- Phase-1 stopline shaping (learn "stop then decide") ---
        self.stop_zone_m = float(getattr(globals(), "STOP_ZONE_M", 1.0))        # meters around stopline
        self.stop_speed_penalty_k = float(getattr(globals(), "STOP_K", 2.0))    # penalty scale
        self.stop_speed_bonus = float(getattr(globals(), "STOP_BONUS", 1.0))    # bonus when stopped in zone

        # ---- hard brake penalty threshold ----
        self.hard_brake_thresh = 3.0
        # --- learning/shaping knobs (tune if needed) ---
        self.D_go_zone = 8.0
        self.go_pos_w = 0.15
        self.go_neg_w = 0.20
        self.go_stop_penalty = 0.05

        # timeout shaping (only when max_steps reached without success)
        self.timeout_penalty_base = 1.0
        self.timeout_penalty_dist_w = 0.05

        self.stall_speed_eps = 0.05

        # ---- over-yield ----
        self.wait_counter = 0
        self.v_stop = 0.15
        self.wait_horizon_steps = int(1.5 / self.delta_seconds)


    def reset(self):
        try:
            self._stop_sensor_listeners()
            self._destroy_episode_actors()

            # episode vars
            self.timesteps = 0
            self.prev_accel = 0.0
            self.collisions_history = []
            self.image_obs = None
            self.navigation_obs = None
            self._u_prev = 0.0  # (not a behavior change; just avoids carry-over from previous episode)

            # scenario params
            self.distance_to_stop_line = float(random.randint(20, 30))
            self.prev_D = self.distance_to_stop_line

            self.ego_has_priority = int(random.random() < 0.5)

            self.start_wp, self.end_wp = self.get_start_end_waypoints(
                self.start_stop_line_wp, self.end_stop_line_wp,
                D_before=self.distance_to_stop_line, exit_ahead=25.0
            )

            # spawn ego
            ego_bp = self.blueprint_library.find(CAR_NAME)
            if ego_bp is None:
                raise RuntimeError(f"Blueprint '{CAR_NAME}' not found.")
            if ego_bp.has_attribute("role_name"):
                ego_bp.set_attribute("role_name", "hero")

            spawn_tf = self.start_wp.transform
            spawn_tf.location.z = max(spawn_tf.location.z + 0.30, 0.30)

            self.ego_vehicle = None
            for dz in (0.30, 0.50, 0.70):
                for dy in (0.00, 0.25, -0.25):
                    tf = carla.Transform(
                        carla.Location(
                            x=spawn_tf.location.x,
                            y=spawn_tf.location.y + dy,
                            z=spawn_tf.location.z + dz
                        ),
                        spawn_tf.rotation
                    )
                    actor = self.world.try_spawn_actor(ego_bp, tf)
                    if actor is not None:
                        self.ego_vehicle = actor
                        break
                if self.ego_vehicle is not None:
                    break

            if self.ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego after nudges.")

            self.actors_list.append(self.ego_vehicle)
            self.ego_vehicle.set_autopilot(False)

            # ---------------- Route ----------------
            self.route_waypoints = self.get_route_waypoints(self.start_wp, self.end_wp, resolution=1.0)
            self.current_waypoint_index = 0


            # PID steering controller
            self.pid_controller = VehiclePIDController(
                self.ego_vehicle,
                args_lateral=self.pid_args_lateral,
                args_longitudinal=self.pid_args_longitudinal
            )

            # start from rest
            try:
                self.ego_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception:
                pass
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0))

            # sensors
            self.camera_obj = CameraSensor(self.ego_vehicle)
            self.sensors_list.append(self.camera_obj.sensor)

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.ego_vehicle)
                self.sensors_list.append(self.env_camera_obj.sensor)

            self.collision_obj = CollisionSensor(self.ego_vehicle)
            self.sensors_list.append(self.collision_obj.sensor)

            # spawn 8 other vehicles via TM
            self.other_vehicles = [None] * self.num_other_agents
            spawned = 0
            for i in range(self.num_other_agents):
                try:
                    veh = self.spawn_crossing_vehicle_tm(i)
                except Exception as e:
                    print(f"[reset] spawn_crossing_vehicle_tm({i}) failed: {e}")
                    veh = None

                self.other_vehicles[i] = veh
                if veh is not None:
                    self.actors_list.append(veh)
                    spawned += 1

            if spawned == 0:
                # if you really want to force traffic existence
                raise RuntimeError("[reset] No other vehicles spawned at all.")


            # TTC
            self.ego_TTC = [float("inf")]
            self.other_TTC = [float("inf")] * self.num_other_agents

            # tick a bit
            for _ in range(2):
                self.world.tick()

            # pull first image if possible
            t0 = time.time()
            while self.image_obs is None and (time.time() - t0) < 2.0:
                self._update_image_obs()
                self.world.tick()

            self.navigation_obs = self._make_navigation_obs(
                ego_speed=0.0,
                D=self.distance_to_stop_line,
                prev_accel=self.prev_accel,
                prio=int(self.ego_has_priority),
                other_speed=0.0,
                ego_ttc=float("inf"),
                other_ttc=float("inf"),
            )

            return [self.image_obs, self.navigation_obs]

        except Exception:
            try:
                self._stop_sensor_listeners()
            except Exception:
                pass
            try:
                self._destroy_episode_actors()
            except Exception:
                pass
            raise

    def step(self, action):
        try:
            # --- action (pedal) ---
            try:
                u_raw = float(np.asarray(action).item())
            except Exception:
                u_raw = float(action)

            if not np.isfinite(u_raw):
                u_raw = 0.0

            u_raw = float(np.clip(u_raw, -1.0, 1.0))

            if abs(u_raw) < self.pedal_deadzone:
                u_raw = 0.0

            du = float(np.clip(u_raw - self._u_prev, -self.pedal_slew, self.pedal_slew))
            u = float(np.clip(self._u_prev + du, -1.0, 1.0))
            self._u_prev = u

            try:
                u_cmd = float(np.asarray(u).item())
            except Exception:
                u_cmd = float(u)
            u_cmd = float(np.clip(u_cmd, -1.0, 1.0))

            if u >= 0.0:
                a_cmd = float(u * self.max_accel)
            else:
                a_cmd = float(u * self.max_decel)  # negative

            # PID steering target
            ego_loc = self.ego_vehicle.get_location()
            ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

            # Prefer global-route target if available, otherwise fallback to lane-follow
            target_wp = self._get_route_target_wp(ego_loc)
            if target_wp is None:
                nxt = ego_wp.next(self.lookahead_dist)
                target_wp = nxt[0] if (nxt and len(nxt) > 0) else ego_wp


            steer = 0.0
            if self.pid_controller is not None:
                pid_control = self.pid_controller.run_step(target_speed=1.0, waypoint=target_wp)
                steer = float(pid_control.steer)

            throttle, brake = (self._pedal_to_control(u) if getattr(self, 'use_pedal_action', False) else self._accel_to_control(a_cmd))
            self.ego_vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, hand_brake=False)
            )

            self.timesteps += 1
            self.world.tick()
            if self.realtime:
                time.sleep(self.delta_seconds)

            # refresh TM paths (keep them straight)
            if (self.timesteps % 40) == 0:
                for v in self.other_vehicles:
                    if v is not None and v.is_alive:
                        self._tm_set_straight_path(v, n_points=5, step_m=25.0)

            # NEW: toggle collision_detection for 50% agents occasionally
            if (self.timesteps % 20) == 0:  # every 1s if dt=0.05
                for i, v in enumerate(self.other_vehicles):
                    if v is None or (not v.is_alive) or (self.ego_vehicle is None):
                        continue
                    try:
                        if int(self.cross_ignore_pct[i]) == 50:
                            toggle_respect = bool(np.random.rand() < 0.5)
                            self.tm.collision_detection(v, self.ego_vehicle, toggle_respect)
                        elif int(self.cross_ignore_pct[i]) >= 100:
                            self.tm.collision_detection(v, self.ego_vehicle, False)
                        else:
                            self.tm.collision_detection(v, self.ego_vehicle, True)
                    except Exception:
                        pass

            self._update_image_obs()

            v = self.ego_vehicle.get_velocity()
            self.ego_current_speed = float(np.linalg.norm([v.x, v.y, v.z]))

            # longitudinal velocity for D update
            fwd = self.ego_vehicle.get_transform().get_forward_vector()
            fwd_xy = np.array([float(fwd.x), float(fwd.y)], dtype=np.float32)
            n = float(np.linalg.norm(fwd_xy)) + 1e-9
            fwd_xy /= n
            v_xy = np.array([float(v.x), float(v.y)], dtype=np.float32)
            v_long = float(np.dot(v_xy, fwd_xy))

            D_old = float(self.distance_to_stop_line)
            D_new = float(D_old - v_long * self.delta_seconds)

            approach = max(0.0, max(D_old, 0.0) - max(D_new, 0.0))
            clearance = max(0.0, max(-D_new, 0.0) - max(-D_old, 0.0))
            dD = float(approach + clearance)

            self.distance_to_stop_line = D_new
            self.prev_D = D_new

            jerk_proxy = float(abs(a_cmd - self.prev_accel))
            self.prev_accel = a_cmd

            self.collisions_history = self.collision_obj.collision_data if self.collision_obj else []
            col_now = (len(self.collisions_history) != 0)
            col_t = 1 if col_now else 0

            # TTC computation TO-DO1
            self.ego_TTC = self.compute_ttc(current_transform = ego_wp, conflict_transform = self.conflict_points_xy[0], speed = self.ego_current_speed)
    
            # Other agent wp
            other_loc = self.other_vehicles[0].get_location()
            other_wp = self.map.get_waypoint(other_loc, project_to_road=True, lane_type=carla.LaneType.Driving)   

            v = self.other_vehicles[0].get_velocity()
            self.other_current_speed = float(np.linalg.norm([v.x, v.y, v.z])) 
            self.other_TTC = self.compute_ttc(current_transform = other_wp, conflict_xy = self.conflict_points_xy[0], speed = self.other_current_speed)

            # ---------- Reward Parameters ---------
            ttc_diff = abs(self.ego_TTC - self.other_TTC)
            # We keep enter_t based on ego crossing stopline (same as before)
            enter_t = 1 if (float(D_old) > 0.0 and float(D_new) <= 0.0) else 0
            # unsafe commit should consider ALL relevant agents
            unsafe_commit = 1 if (enter_t == 1 and ttc_diff < self.ttc_critical) else 0

            prio = int(self.ego_has_priority)

            safe_t = 1 if (ttc_diff >= self.ttc_safe) else 0

            stopped_t = 1 if (self.ego_current_speed < self.v_stop) else 0

            if prio == 1 and safe_t == 1 and stopped_t == 1:
                self.wait_counter += 1
            else:
                self.wait_counter = 0

            over_yield = 1 if (self.wait_counter >= self.wait_horizon_steps) else 0
            wait_ratio = float(min(1.0, self.wait_counter / max(1, self.wait_horizon_steps)))

            # ---------- Reward Formulation ----------

            r_progress = +1.0 * dD
            r_time = -0.01

            r_smooth = -0.05 * jerk_proxy
            r_hardbrake = -0.10 * (1 if a_cmd < -self.hard_brake_thresh else 0)

            r_safety = 0.0
            if unsafe_commit == 1:
                r_safety -= 2.0

            if col_t:
                r_safety -= 10.0

            r_neg = 0.0
            if over_yield:
                r_neg -= 0.3

            # ---------- Stopline "notice intersection" shaping ----------
            stop_zone = 1 if (abs(float(D_new)) <= self.stop_zone_m) else 0

            r_stopline = 0.0
            if stop_zone:
                # penalize moving at the stopline (forces brake-to-zero behavior)
                r_stopline -= self.stop_speed_penalty_k * float(self.ego_current_speed)

                # small bonus for actually being stopped in the stop zone
                if stopped_t == 1:
                    r_stopline += self.stop_speed_bonus


            reward = (
                self.weight_of_efficiency * (r_progress + r_time)
                + self.weight_of_smoothness * (r_smooth + r_hardbrake)
                + self.weight_of_safety * r_safety
                + self.weight_of_negotiation_quality * r_neg
                + r_stopline
            )


            # --- 'go when safe' shaping near stop line ---
            if (prio == 1) and (safe_t == 1) and (D_new <= self.D_go_zone):
                reward += self.go_pos_w * max(0.0, u_cmd)
                reward -= self.go_neg_w * max(0.0, -u_cmd)
                if self.ego_current_speed < self.stall_speed_eps:
                    reward -= self.go_stop_penalty
                

            done = False
            success = False
            timed_out = False

            if col_t:
                done = True

            # strict success: cleared stopline by success_clearance
            if D_new <= -self.success_clearance: 
                success = True
                if self.count_s == 0:
                    reward += 5.0
                self.count_s += 1


            # curriculum success flag (optional): crossing stopline is "success", but DO NOT end episode here
            if (not success) and (prio == 1) and (safe_t == 1) and (D_new <= 0.0):
                success = True

            # --- true termination: reaching the end waypoint of the route ---
            goal_dist = float("inf")
            try:
                if self.end_wp is not None:
                    ego_loc = self.ego_vehicle.get_transform().location
                    goal_dist = float(ego_loc.distance(self.end_wp.transform.location))
            except Exception:
                pass

            if (not done) and (goal_dist <= self.success_clearance):
                success = True
                done = True

            # episode timeout (max_steps)
            if (not done) and (self.timesteps >= self.max_steps):
                done = True
                timed_out = True
                if not success:
                    reward -= (
                        self.timeout_penalty_base
                        + self.timeout_penalty_dist_w * max(0.0, float(D_new))
                    )

            self.navigation_obs = self._make_navigation_obs(
                ego_speed=0.0,
                D=self.distance_to_stop_line,
                prev_accel=self.prev_accel,
                prio=int(self.ego_has_priority),
                other_speed=0.0,
                ego_ttc=float("inf"),
                other_ttc=float("inf"),
            )

            info = {
                "dD": float(dD),
                "jerk_proxy": float(jerk_proxy),
                "a_cmd": float(a_cmd),
                "u_cmd": float(u_cmd),

                "prio": int(prio),
                "over_yield": int(over_yield),

                "enter_t": int(enter_t),
                "unsafe_commit": int(unsafe_commit),
                "collision": int(col_t),

                # termination/debug flags
                "timed_out": int(timed_out),
            
                "success": int(success),
                "D": float(D_new),
                "ego TTC": float(self.ego_TTC if np.isfinite(self.ego_TTC) else 1e6),
                "other TTC": float(self.other_TTC if np.isfinite(self.other_TTC) else 1e6),
                "stop_zone": int(stop_zone),
                "r_stopline": float(r_stopline),
            }

            return [self.image_obs, self.navigation_obs], float(reward), bool(done), info

        except Exception as e:
            info = {"error": str(e)}
            return [getattr(self, "image_obs", None),
                    getattr(self, "navigation_obs", None)], 0.0, True, info