#!/usr/bin/python3
"""
Phase 2 — GPS-denied autonomous exploration mission (ASCEND IRoC-U 2026).

State machine:
  IDLE → PRESTREAM → TAKEOFF → EXPLORE → RETURN → DESCEND → LAND → DONE
  Any active phase → STABILIZE → (resume saved phase)  on attitude fault

Exploration: boustrophedon (lawnmower) coverage of the full arena.

Wall-safety (3-layer):
  Layer 1 — Goal clamping:  all EXPLORE goals clamped to ≥ WALL_STOP_MARGIN
             from actual walls. Waypoints are still generated at the geofence
             edges for coverage math, but the effective commanded stop is 1.5 m
             inside each wall.
  Layer 2 — Brake zone:     carrot speed reduces linearly from MAX_XY_SPEED
             at WALL_BRAKE_START down to WALL_BRAKE_MIN at WALL_STOP_MARGIN,
             then to 0. Prevents momentum overshoot regardless of PX4 lag.
  Layer 3 — Hard freeze:    if carrot ever reaches WALL_HARD_LIMIT from any
             wall (e.g. from compounding errors), it is clamped in place.
             If the drone's actual position breaches WALL_HARD_LIMIT, the
             current waypoint is abandoned and the next one issued immediately.

Camera model:
  The camera rotates among tilt angles 45°, 65°, 90° from nadir during survey.
    45° tilt at H=3 m cruise → ground lookahead = H × tan(45°) = 3.0 m
    65° tilt at H=3 m cruise → ground lookahead = H × tan(65°) ≈ 6.4 m
    90° (horizontal)         → direct wall detection
  CAM_LOOKAHEAD_M uses the conservative 45° angle (3.0 m).
  _camera_sees_wall() simulates what the camera would detect by position math.
  Replace its internals with real image-pipeline output for hardware flight.

LOCAL frame ≈ Gazebo world ENU (EKF2_EV_CTRL=15, absolute EV fusion).  Base station at world/LOCAL ≈ (0.80, 0.80).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                        ReliabilityPolicy)
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# ── Flight parameters ──────────────────────────────────────────────────────
FLIGHT_ALT   = 3.0    # m  cruise altitude (within 2–6 m band)
DESCENT_ALT  = 1.2    # m  altitude before AUTO.LAND fires
WP_RADIUS    = 0.45   # m  waypoint acceptance sphere
BASE_RADIUS  = 0.30   # m  tighter acceptance for return-to-base
SETPOINT_HZ  = 20.0   # Hz
PRESTREAM_S  = 2.0    # s  pre-stream before OFFBOARD switch

MAX_XY_SPEED = 1.5    # m/s  lateral cruise speed
MAX_Z_SPEED  = 0.8    # m/s  vertical speed

WP_TIMEOUT   = 30.0   # s  abandon waypoint if not reached in time

# ── Lawnmower waypoint grid (LOCAL ≈ world ENU) ────────────────────────────
# Arena world: (0,0)→(10.668,7.620).  LOCAL ≈ world ENU (absolute EV fusion).
# Grid edges 0.5 m inside tape for full coverage; Layer 1 clamps the *commanded*
# goal further inward to WALL_STOP_MARGIN.
GEO_XMIN =  0.50
GEO_XMAX = 10.17
GEO_YMIN =  0.50
GEO_YMAX =  7.12

STRIP_WIDTH = 2.0   # m  N–S spacing between E–W survey strips

# ── Tape boundary positions in LOCAL ≈ world ENU ──────────────────────────
# EKF2 absolute EV fusion → LOCAL ≈ Gazebo world frame.
WALL_X_MIN =  0.00    # west tape   (world x = 0.00)
WALL_X_MAX = 10.668   # east tape   (world x = 10.668)
WALL_Y_MIN =  0.00    # south tape  (world y = 0.00)
WALL_Y_MAX =  7.620   # north tape  (world y = 7.620)

# ── Camera-based boundary detection constants ──────────────────────────────
# Camera rotates to 45°, 65°, 90° from nadir during survey.
# Conservative working angle = 45°  →  lookahead = FLIGHT_ALT = 3.0 m.
CAM_LOOKAHEAD_M  = 3.0   # m  camera detects wall this far ahead (45° tilt)

WALL_STOP_MARGIN = 1.5   # m  target stop distance from actual wall (Layer 1)
# WALL_BRAKE_START = 2.5 m gives exactly 1.0 m of braking distance
# (BRAKE_START − STOP_MARGIN = 2.5 − 1.5 = 1.0 m).  At MAX_XY_SPEED=1.5 m/s
# this is ~1.3 s of deceleration — sufficient for PX4's velocity controller.
# Using 4.0 m caused the entire last-step-north leg (1.62 m) to be in the
# brake zone, making the last stride visibly slow.
WALL_BRAKE_START = 2.5   # m  begin speed reduction at this dist from wall (Layer 2)
WALL_BRAKE_MIN   = 0.25  # m/s  minimum carrot speed inside brake zone
WALL_HARD_LIMIT  = 0.5   # m  carrot freeze / emergency repath threshold (Layer 3)
# Must be < drone's spawn wall distances (spawn at LOCAL ≈ world (0.80,0.80):
# west=0.80 m, south=0.80 m).  0.5 m catches crashes without false triggers at base.

# ── Base station (landing target) in LOCAL ≈ world ENU ────────────────────
# Pad at world/LOCAL ≈ (0.80, 0.80). Drone spawns on pad.
# 0.80 m from west/south boundary tape (> RETURN_WALL_STOP=0.25 m).
BASE_X = 0.80  # m  LOCAL ≈ world east  (base station world x)
BASE_Y = 0.80  # m  LOCAL ≈ world north (base station world y)

# ── RETURN phase wall limits ───────────────────────────────────────────────
# BASE at (0.80,0.80) is 0.80 m from west/south tape — reachable, won't trigger stop.
RETURN_WALL_STOP  = 0.25  # m    stop this close to wall during RETURN/DESCEND
RETURN_WALL_BRAKE = 3.0   # m    start braking at this distance from wall
RETURN_MAX_SPEED  = 0.8   # m/s  cap for all RETURN/DESCEND/TAKEOFF nav

# ── Staged return path ─────────────────────────────────────────────────────
# RETURN follows: (optional) arena-center → base (BASE_X, BASE_Y).
# Center hop avoids the NW→SW wall-adjacent diagonal that caused crashes.
RETURN_VIA_RADIUS  = 0.6   # m    acceptance radius for intermediate return WPs
RETURN_FINAL_SPEED = 0.3   # m/s  speed cap for the last leg to the landing pad

# ── Stability / attitude recovery ──────────────────────────────────────────
# If roll or pitch exceeds UNSTABLE_ANGLE_DEG, the drone enters STABILIZE:
# carrot is frozen at current position, PX4 recovers attitude, then the
# saved phase is resumed. Prevents tumbling from becoming a total loss.
UNSTABLE_ANGLE_DEG = 35.0  # deg  enter STABILIZE above this tilt
STABLE_ANGLE_DEG   = 10.0  # deg  must be below this to exit STABILIZE
STABLE_HOLD_S      = 1.5   # s    must hold stable before resuming

# ── Derived safe goal bounds (Layer 1) ─────────────────────────────────────
# Base (0.80,0.80) is outside safe zone but RETURN/DESCEND use clamp=False.
_SAFE_XMIN = WALL_X_MIN + WALL_STOP_MARGIN   #  1.50 m
_SAFE_XMAX = WALL_X_MAX - WALL_STOP_MARGIN   #  9.168 m
_SAFE_YMIN = WALL_Y_MIN + WALL_STOP_MARGIN   #  1.50 m
_SAFE_YMAX = WALL_Y_MAX - WALL_STOP_MARGIN   #  6.120 m


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_lawnmower():
    """
    Boustrophedon (E–W strips, stepping N) in LOCAL ENU.

    Waypoints are at the geofence grid edges (good for coverage math).
    _set_goal(clamp=True) remaps them to the safe zone at runtime.

    Generated WPs (LOCAL ≈ world ENU):
      (10.17,0.50) → (10.17,2.50) → (0.50,2.50) → (0.50,4.50)
      → (10.17,4.50) → (10.17,6.50) → (0.50,6.50)
    After Layer-1 clamp to safe zone (1.50→9.168, 1.50→6.12):
      (9.17,1.50)  → (9.17,2.50)  → (1.50,2.50)  → (1.50,4.50)
      → (9.17,4.50) → (9.17,6.12) → (1.50,6.12)
    """
    wps = []
    y = GEO_YMIN
    going_east = True

    while y <= GEO_YMAX + 1e-6:
        x = GEO_XMAX if going_east else GEO_XMIN
        wps.append((x, y, FLIGHT_ALT))

        y_next = y + STRIP_WIDTH
        if y_next <= GEO_YMAX + 1e-6:
            wps.append((x, min(y_next, GEO_YMAX), FLIGHT_ALT))
            y = y_next
            going_east = not going_east
        else:
            break

    return wps


class ExplorationMission(Node):

    def __init__(self):
        super().__init__('exploration_mission')

        cbg = ReentrantCallbackGroup()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(State, '/mavros/state',
                                 self._state_cb, qos, callback_group=cbg)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._pose_cb, qos, callback_group=cbg)

        self._sp_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', qos)

        self._arm_cli  = self.create_client(CommandBool, '/mavros/cmd/arming',
                                            callback_group=cbg)
        self._mode_cli = self.create_client(SetMode, '/mavros/set_mode',
                                            callback_group=cbg)

        self._mav_state = State()
        self._pose      = PoseStamped()
        self._phase     = 'IDLE'
        self._wp_idx    = 0
        self._waypoints = build_lawnmower()
        self._t_phase   = time.time()

        self._goal        = (0.0, 0.0, FLIGHT_ALT)
        self._sp          = self._make_sp(0.0, 0.0, 0.3, clamp=False)
        self._wp_deadline = float('inf')

        self._prestream_done      = False
        self._last_offboard_req_t = 0.0
        self._brake_logged        = False

        # Stability recovery state
        self._stabilize_return_to = None          # phase to resume after STABILIZE
        self._stable_since        = None          # time attitude first recovered
        self._last_safe_pos       = (0.0, 0.0, FLIGHT_ALT)  # last pos w/ stable attitude

        # Staged return path
        self._return_wps    : list = []   # waypoints built at RETURN start
        self._return_wp_idx : int  = 0    # current index into _return_wps

        self.get_logger().info(
            f'Exploration mission ready. {len(self._waypoints)} waypoints.')
        self.get_logger().info(
            f'WP grid: x[{GEO_XMIN},{GEO_XMAX}] y[{GEO_YMIN},{GEO_YMAX}]')
        self.get_logger().info(
            f'Safe zone (Layer 1): x[{_SAFE_XMIN:.2f},{_SAFE_XMAX:.2f}] '
            f'y[{_SAFE_YMIN:.2f},{_SAFE_YMAX:.2f}]')
        self.get_logger().info(
            f'Wall brake: start={WALL_BRAKE_START}m  stop={WALL_STOP_MARGIN}m  '
            f'hard={WALL_HARD_LIMIT}m  cam_lookahead={CAM_LOOKAHEAD_M}m')
        self.get_logger().info('Raw waypoints: ' + str(self._waypoints))

        self._timer = self.create_timer(1.0 / SETPOINT_HZ, self._loop,
                                        callback_group=cbg)

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _state_cb(self, msg: State):
        self._mav_state = msg

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg

    # ── Helpers ────────────────────────────────────────────────────────────
    def _make_sp(self, x: float, y: float, z: float,
                 clamp: bool = True) -> PoseStamped:
        if clamp:
            x = _clamp(x, _SAFE_XMIN, _SAFE_XMAX)
            y = _clamp(y, _SAFE_YMIN, _SAFE_YMAX)
        sp = PoseStamped()
        sp.header.frame_id = 'map'
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z
        sp.pose.orientation.w = 1.0
        return sp

    def _xyz(self):
        p = self._pose.pose.position
        return p.x, p.y, p.z

    def _dist_xyz(self, tx, ty, tz):
        px, py, pz = self._xyz()
        return math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)

    def _wait_future(self, fut, timeout=3.0) -> bool:
        deadline = time.time() + timeout
        while not fut.done() and time.time() < deadline:
            time.sleep(0.05)
        return fut.done()

    def _set_mode(self, mode: str) -> bool:
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self._mode_cli.call_async(req)
        self._wait_future(fut)
        ok = fut.done() and fut.result() is not None and fut.result().mode_sent
        self.get_logger().info(f'set_mode({mode}) → {"OK" if ok else "FAIL"}')
        return ok

    def _arm(self, value: bool) -> bool:
        req = CommandBool.Request()
        req.value = value
        fut = self._arm_cli.call_async(req)
        self._wait_future(fut)
        ok = fut.done() and fut.result() is not None and fut.result().success
        self.get_logger().info(f'arm({value}) → {"OK" if ok else "FAIL"}')
        return ok

    def _set_goal(self, x: float, y: float, z: float, clamp: bool = True):
        """Set navigation goal. clamp=True applies Layer-1 safe bounds."""
        if clamp:
            x = _clamp(x, _SAFE_XMIN, _SAFE_XMAX)
            y = _clamp(y, _SAFE_YMIN, _SAFE_YMAX)
        self._goal = (x, y, z)

    # ── Wall-safety helpers ────────────────────────────────────────────────
    def _wall_distances(self, px: float, py: float) -> dict:
        """Distance from (px,py) to each of the 4 arena walls."""
        return {
            'west':  px - WALL_X_MIN,
            'east':  WALL_X_MAX - px,
            'south': py - WALL_Y_MIN,
            'north': WALL_Y_MAX - py,
        }

    def _roll_pitch_deg(self):
        """Return (|roll|, |pitch|) in degrees from current pose quaternion."""
        q = self._pose.pose.orientation
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            return 0.0, 0.0
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.degrees(math.atan2(sinr, cosr))
        sinp = _clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
        pitch = math.degrees(math.asin(sinp))
        return abs(roll), abs(pitch)

    def _wall_speed_limit(self, px: float, py: float,
                          dx_n: float, dy_n: float) -> float:
        """
        Layer-2 brake zone: return carrot speed cap based on distance to any
        wall the drone is currently approaching.

        Linear ramp: full speed at WALL_BRAKE_START, WALL_BRAKE_MIN at
        WALL_STOP_MARGIN, 0 at or inside WALL_STOP_MARGIN.
        Only walls in the direction of travel are considered.
        """
        d = self._wall_distances(px, py)
        min_approach = float('inf')

        if dx_n >  0.01:  min_approach = min(min_approach, d['east'])
        if dx_n < -0.01:  min_approach = min(min_approach, d['west'])
        if dy_n >  0.01:  min_approach = min(min_approach, d['north'])
        if dy_n < -0.01:  min_approach = min(min_approach, d['south'])

        if min_approach <= WALL_STOP_MARGIN:
            return 0.0
        if min_approach <= WALL_BRAKE_START:
            t = (min_approach - WALL_STOP_MARGIN) / (WALL_BRAKE_START - WALL_STOP_MARGIN)
            return WALL_BRAKE_MIN + t * (MAX_XY_SPEED - WALL_BRAKE_MIN)
        return MAX_XY_SPEED

    def _wall_speed_limit_return(self, px: float, py: float,
                                dx_n: float, dy_n: float) -> float:
        """
        Gentler wall speed limit for RETURN/DESCEND.
        RETURN_WALL_STOP=0.25 m < base-station wall distances (0.80 m),
        so (0,0) remains reachable while preventing high-speed wall crashes.
        """
        d = self._wall_distances(px, py)
        min_approach = float('inf')
        if dx_n >  0.01: min_approach = min(min_approach, d['east'])
        if dx_n < -0.01: min_approach = min(min_approach, d['west'])
        if dy_n >  0.01: min_approach = min(min_approach, d['north'])
        if dy_n < -0.01: min_approach = min(min_approach, d['south'])

        if min_approach <= RETURN_WALL_STOP:
            return 0.0
        if min_approach <= RETURN_WALL_BRAKE:
            t = (min_approach - RETURN_WALL_STOP) / (RETURN_WALL_BRAKE - RETURN_WALL_STOP)
            return WALL_BRAKE_MIN + t * (RETURN_MAX_SPEED - WALL_BRAKE_MIN)
        return RETURN_MAX_SPEED

    def _build_return_path(self, px: float, py: float) -> list:
        """
        Staged return path from (px,py) → landing pad (BASE_X, BASE_Y).

        If starting from the northern half (py > 3.5), goes via arena centre
        first to avoid the NW→SW diagonal that caused repeated wall crashes.
        Arena centre LOCAL/world ≈ (5.3, 3.8).
        """
        wps = []
        if py > 3.5:
            wps.append((5.0, 3.5, FLIGHT_ALT))   # arena centre
        wps.append((BASE_X, BASE_Y, FLIGHT_ALT))  # landing pad
        return wps

    def _camera_sees_wall(self, px: float, py: float,
                          dx_n: float, dy_n: float) -> bool:
        """
        Simulated camera detection: returns True when the drone's camera
        (45° tilt → CAM_LOOKAHEAD_M forward coverage) would first see a wall
        that is within WALL_STOP_MARGIN of the stopping point.

        Trigger condition: dist_to_wall ≤ WALL_STOP_MARGIN + CAM_LOOKAHEAD_M
        i.e. the wall enters the camera's forward FOV.

        Replace the body of this function with real image-pipeline output
        when flying on hardware.  The interface (position + unit direction in,
        bool out) stays the same.
        """
        d = self._wall_distances(px, py)
        detect_thresh = WALL_STOP_MARGIN + CAM_LOOKAHEAD_M  # 4.5 m
        if dx_n >  0.01 and d['east']  <= detect_thresh:  return True
        if dx_n < -0.01 and d['west']  <= detect_thresh:  return True
        if dy_n >  0.01 and d['north'] <= detect_thresh:  return True
        if dy_n < -0.01 and d['south'] <= detect_thresh:  return True
        return False

    def _within_hard_limit(self, px: float, py: float) -> bool:
        """True if position violates WALL_HARD_LIMIT from any wall."""
        d = self._wall_distances(px, py)
        return any(v < WALL_HARD_LIMIT for v in d.values())

    # ── Carrot-chase setpoint ──────────────────────────────────────────────
    def _step_sp_toward_goal(self):
        """
        Advance setpoint carrot one tick toward goal.

        Integrates Layer-2 (speed brake) and Layer-3 (hard freeze).
        Layer-1 is applied at goal-set time via _set_goal(clamp=True).
        """
        gx, gy, gz = self._goal
        dt = 1.0 / SETPOINT_HZ
        px = self._sp.pose.position.x
        py = self._sp.pose.position.y
        pz = self._sp.pose.position.z

        dx, dy = gx - px, gy - py
        dist_xy = math.sqrt(dx * dx + dy * dy)

        if dist_xy > 1e-3:
            dx_n, dy_n = dx / dist_xy, dy / dist_xy

            # Layers 2 & 3 apply only during EXPLORE.
            # RETURN/DESCEND navigate to base station (0,0) which is 0.35 m from
            # two walls — applying wall limits there would freeze the carrot.
            if self._phase == 'EXPLORE':
                spd = self._wall_speed_limit(px, py, dx_n, dy_n)

                if spd < MAX_XY_SPEED * 0.95:
                    if not self._brake_logged:
                        d = self._wall_distances(px, py)
                        self.get_logger().warn(
                            f'CAM: wall in range — brake {spd:.2f} m/s '
                            f'(E:{d["east"]:.1f} W:{d["west"]:.1f} '
                            f'N:{d["north"]:.1f} S:{d["south"]:.1f})')
                        self._brake_logged = True
                else:
                    self._brake_logged = False

                step = min(dist_xy, spd * dt)
                nx = px + dx_n * step
                ny = py + dy_n * step

                # Layer 3: hard freeze during EXPLORE only
                if self._within_hard_limit(nx, ny):
                    nx = _clamp(nx,
                                WALL_X_MIN + WALL_HARD_LIMIT,
                                WALL_X_MAX - WALL_HARD_LIMIT)
                    ny = _clamp(ny,
                                WALL_Y_MIN + WALL_HARD_LIMIT,
                                WALL_Y_MAX - WALL_HARD_LIMIT)
                    self.get_logger().error(
                        f'Layer-3 hard freeze: carrot clamped at '
                        f'({nx:.2f},{ny:.2f})')
            else:
                # RETURN/DESCEND/TAKEOFF: softer wall limits prevent overshoot
                # into SW corner walls; RETURN_WALL_STOP=0.25 m < base (0.35 m)
                # so base station remains reachable.
                spd = self._wall_speed_limit_return(px, py, dx_n, dy_n)
                # Final return leg (to base corner): hard cap at RETURN_FINAL_SPEED
                if (self._phase == 'RETURN' and self._return_wps and
                        self._return_wp_idx == len(self._return_wps) - 1):
                    spd = min(spd, RETURN_FINAL_SPEED)
                step = min(dist_xy, spd * dt)
                nx = px + dx_n * step
                ny = py + dy_n * step

            px, py = nx, ny
        else:
            px, py = gx, gy

        dz = gz - pz
        if abs(dz) > 1e-3:
            pz += math.copysign(min(abs(dz), MAX_Z_SPEED * dt), dz)
        else:
            pz = gz

        self._sp = self._make_sp(px, py, pz, clamp=False)

    def _dist_goal(self) -> float:
        gx, gy, gz = self._goal
        return self._dist_xyz(gx, gy, gz)

    # ── Waypoint advancement ───────────────────────────────────────────────
    def _advance_waypoint(self):
        """
        Move to the next lawnmower waypoint, or transition to RETURN.
        Goals are clamped to safe zone (Layer 1) automatically.
        Logs the effective clamped goal so the operator can verify.
        """
        self._wp_idx += 1
        if self._wp_idx >= len(self._waypoints):
            self.get_logger().info('Lawnmower complete. Returning to base.')
            px, py, _ = self._xyz()
            self._return_wps    = self._build_return_path(px, py)
            self._return_wp_idx = 0
            rx, ry, rz = self._return_wps[0]
            self.get_logger().info(
                f'Return path ({len(self._return_wps)} legs): '
                + ' → '.join(f'({w[0]:.1f},{w[1]:.1f})' for w in self._return_wps))
            self._set_goal(rx, ry, rz, clamp=False)
            self._phase = 'RETURN'
            return

        wx, wy, wz = self._waypoints[self._wp_idx]
        self._set_goal(wx, wy, wz, clamp=True)
        self._wp_deadline = time.time() + WP_TIMEOUT
        gx, gy, _ = self._goal
        self.get_logger().info(
            f'WP {self._wp_idx}/{len(self._waypoints)}: '
            f'raw=({wx:.2f},{wy:.2f}) → clamped=({gx:.2f},{gy:.2f})')

    # ── Main control loop (20 Hz) ──────────────────────────────────────────
    def _loop(self):
        self._step_sp_toward_goal()
        self._sp.header.stamp = self.get_clock().now().to_msg()
        self._sp_pub.publish(self._sp)

        # ── Stability monitor ──────────────────────────────────────────────
        # Runs every tick in all active flight phases. If roll or pitch
        # exceeds UNSTABLE_ANGLE_DEG, enter STABILIZE: freeze carrot at
        # current position and wait for PX4 to recover attitude before resuming.
        if self._phase not in ('IDLE', 'PRESTREAM', 'LAND', 'DONE', 'STABILIZE'):
            px, py, pz = self._xyz()
            roll, pitch = self._roll_pitch_deg()
            # Skip during low-altitude TAKEOFF: ground contact and motor
            # spin-up cause real oscillations that are not in-flight crashes.
            takeoff_near_ground = (self._phase == 'TAKEOFF' and pz < 1.5)
            if not takeoff_near_ground and \
                    (roll > UNSTABLE_ANGLE_DEG or pitch > UNSTABLE_ANGLE_DEG):
                sx, sy, sz = self._last_safe_pos
                self.get_logger().error(
                    f'UNSTABLE: roll={roll:.1f}° pitch={pitch:.1f}° at '
                    f'({px:.2f},{py:.2f},{pz:.2f}) — recovering to safe '
                    f'({sx:.2f},{sy:.2f},{sz:.2f}), was {self._phase}')
                self._stabilize_return_to = self._phase
                self._set_goal(sx, sy, sz, clamp=False)
                self._stable_since = None
                self._phase = 'STABILIZE'
                return
            else:
                # Track last known safe position (stable attitude, inside arena)
                px, py, pz = self._xyz()
                if (WALL_X_MIN + 0.5 <= px <= WALL_X_MAX - 0.5 and
                        WALL_Y_MIN + 0.5 <= py <= WALL_Y_MAX - 0.5):
                    self._last_safe_pos = (px, py, max(pz, FLIGHT_ALT))

        # OFFBOARD watchdog
        if (self._phase not in ('IDLE', 'PRESTREAM', 'LAND', 'DONE') and
                self._mav_state.armed and
                self._mav_state.mode != 'OFFBOARD' and
                time.time() - self._last_offboard_req_t > 1.0):
            self._last_offboard_req_t = time.time()
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            self._mode_cli.call_async(req)
            self.get_logger().warn(
                f'OFFBOARD lost ({self._mav_state.mode}), re-requesting…')

        # ── IDLE ──
        if self._phase == 'IDLE':
            if not self._mav_state.connected:
                return
            self.get_logger().info('MAVROS connected. Pre-streaming setpoints…')
            self._phase   = 'PRESTREAM'
            self._t_phase = time.time()

        # ── PRESTREAM ──
        elif self._phase == 'PRESTREAM':
            if time.time() - self._t_phase < PRESTREAM_S:
                return
            if self._prestream_done:
                return
            self._prestream_done = True
            if not self._set_mode('OFFBOARD'):
                self._t_phase = time.time()
                return
            if not self._arm(True):
                self._prestream_done = False
                self._t_phase = time.time()
                return
            self.get_logger().info(f'Armed! Climbing to {FLIGHT_ALT} m…')
            self._set_goal(0.0, 0.0, FLIGHT_ALT, clamp=False)
            self._phase = 'TAKEOFF'

        # ── TAKEOFF ──
        elif self._phase == 'TAKEOFF':
            px, py, z = self._xyz()
            elapsed = time.time() - self._t_phase
            self._goal = (px, py, FLIGHT_ALT)  # hold XY during climb
            if z >= FLIGHT_ALT - 0.30:
                self.get_logger().info(f'At {z:.2f} m. Starting exploration.')
                self._wp_idx = 0
                wx, wy, wz = self._waypoints[0]
                self._set_goal(wx, wy, wz, clamp=True)
                self._wp_deadline = time.time() + WP_TIMEOUT
                self._phase = 'EXPLORE'
                gx, gy, _ = self._goal
                self.get_logger().info(
                    f'WP 0/{len(self._waypoints)}: '
                    f'raw=({wx:.2f},{wy:.2f}) → clamped=({gx:.2f},{gy:.2f})')
            elif elapsed > 20.0 and z < 0.5:
                self.get_logger().error(
                    f'TAKEOFF abort: z={z:.2f}m after 20 s. Landing.')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

        # ── EXPLORE ──
        elif self._phase == 'EXPLORE':
            px, py, pz = self._xyz()

            # Layer-3 emergency: drone physically breached hard limit → repath now
            if self._within_hard_limit(px, py):
                d = self._wall_distances(px, py)
                self.get_logger().error(
                    f'EMERGENCY: drone at ({px:.2f},{py:.2f}) breached '
                    f'hard limit {WALL_HARD_LIMIT}m '
                    f'(E:{d["east"]:.2f} W:{d["west"]:.2f} '
                    f'N:{d["north"]:.2f} S:{d["south"]:.2f}) — repaths')
                self._advance_waypoint()
                return

            # Camera lookahead: log when camera would first detect approaching wall.
            # Drone position (not carrot) is used — camera is on the drone.
            gx, gy, _ = self._goal
            dx, dy = gx - px, gy - py
            dist_to_goal = math.sqrt(dx*dx + dy*dy)
            if dist_to_goal > 0.2:
                dx_n, dy_n = dx / dist_to_goal, dy / dist_to_goal
                if self._camera_sees_wall(px, py, dx_n, dy_n):
                    d = self._wall_distances(px, py)
                    self.get_logger().info(
                        f'CAM: wall in FOV at ({px:.2f},{py:.2f}) '
                        f'dir=({dx_n:.1f},{dy_n:.1f}) '
                        f'E:{d["east"]:.1f} W:{d["west"]:.1f} '
                        f'N:{d["north"]:.1f} S:{d["south"]:.1f} — braking…')

            # Normal WP acceptance (or timeout)
            timed_out = time.time() > self._wp_deadline
            if self._dist_goal() < WP_RADIUS or timed_out:
                if timed_out:
                    self.get_logger().warn(
                        f'WP {self._wp_idx} timed out after {WP_TIMEOUT} s.')
                self._advance_waypoint()

        # ── RETURN ──
        elif self._phase == 'RETURN':
            is_last = (self._return_wp_idx == len(self._return_wps) - 1)
            radius = BASE_RADIUS if is_last else RETURN_VIA_RADIUS
            if self._dist_goal() < radius:
                self._return_wp_idx += 1
                if self._return_wp_idx >= len(self._return_wps):
                    self.get_logger().info(
                        f'Above base. Descending to {DESCENT_ALT} m.')
                    self._set_goal(BASE_X, BASE_Y, DESCENT_ALT, clamp=False)
                    self._phase = 'DESCEND'
                else:
                    rx, ry, rz = self._return_wps[self._return_wp_idx]
                    self.get_logger().info(
                        f'Return leg {self._return_wp_idx}/{len(self._return_wps)}: '
                        f'→ ({rx:.1f},{ry:.1f})')
                    self._set_goal(rx, ry, rz, clamp=False)

        # ── DESCEND ──
        elif self._phase == 'DESCEND':
            _, _, z = self._xyz()
            if z <= DESCENT_ALT + 0.20:
                self.get_logger().info('Triggering AUTO.LAND.')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

        # ── STABILIZE ──
        elif self._phase == 'STABILIZE':
            roll, pitch = self._roll_pitch_deg()
            # Goal is already fixed at _last_safe_pos (set at trigger time).
            # Do NOT overwrite it here — that caused recovery to command the
            # crashed/outside-arena position each tick.

            if roll < STABLE_ANGLE_DEG and pitch < STABLE_ANGLE_DEG:
                if self._stable_since is None:
                    self._stable_since = time.time()
                    self.get_logger().info(
                        f'Attitude normalising (roll={roll:.1f}° pitch={pitch:.1f}°) '
                        f'— holding for {STABLE_HOLD_S} s before resume…')
                elif time.time() - self._stable_since >= STABLE_HOLD_S:
                    self.get_logger().info(
                        f'Attitude stable — resuming {self._stabilize_return_to}')
                    self._phase = self._stabilize_return_to
                    # Re-issue the correct goal for the resumed phase
                    if self._stabilize_return_to == 'EXPLORE':
                        wx, wy, wz = self._waypoints[self._wp_idx]
                        self._set_goal(wx, wy, wz, clamp=True)
                        self._wp_deadline = time.time() + WP_TIMEOUT
                    elif self._stabilize_return_to == 'RETURN':
                        if self._return_wps and self._return_wp_idx < len(self._return_wps):
                            rx, ry, rz = self._return_wps[self._return_wp_idx]
                            self._set_goal(rx, ry, rz, clamp=False)
                        else:
                            self._set_goal(BASE_X, BASE_Y, FLIGHT_ALT, clamp=False)
                    elif self._stabilize_return_to == 'DESCEND':
                        self._set_goal(BASE_X, BASE_Y, DESCENT_ALT, clamp=False)
            else:
                self._stable_since = None  # reset timer — not stable yet

        # ── LAND ──
        elif self._phase == 'LAND':
            if not self._mav_state.armed:
                self.get_logger().info('Landed and disarmed. Mission complete.')
                self._phase = 'DONE'
                raise SystemExit(0)


def main():
    rclpy.init()
    node = ExplorationMission()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
