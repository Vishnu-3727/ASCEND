#!/usr/bin/python3
"""
Phase 2A — GPS-denied autonomous exploration, polygon-based (ASCEND IRoC-U 2026).

State machine (identical sequencing to exploration_mission.py):
  IDLE → PRESTREAM → TAKEOFF → EXPLORE → RETURN → DESCEND → LAND → DONE
  Any active phase → STABILIZE → (resume saved phase)  on attitude fault

Exploration: frontier-based grid coverage within an arbitrary polygon arena.
  Change ARENA_POLYGON to test any shape — no other code changes needed.

Safety layers (polygon-aware):
  Layer 1 — Waypoint gating:  all EXPLORE waypoints are inside inner_polygon
             before they are ever set as goals.
  Layer 2 — Brake zone:       carrot speed reduced near polygon boundary.
  Layer 3 — Hard freeze:      carrot frozen if it would exit inner_polygon.

LOCAL frame = EKF2 local frame starting at arm point (EKF2_EV_CTRL=0, pure optical flow).
Base station at EKF origin (0.0, 0.0); Gazebo world spawn is (0.80, 0.80).

Do NOT modify src/exploration_mission.py — this is a parallel Phase 2A file.
"""

import heapq
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                        ReliabilityPolicy)
from geometry_msgs.msg import PoseStamped, PointStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# ── Flight parameters (kept identical to exploration_mission.py) ───────────
FLIGHT_ALT    = 1.5     # m  lower = better optical flow SNR in SITL
DESCENT_ALT   = 0.8     # m  altitude before AUTO.LAND fires
WP_RADIUS     = 0.45    # m  waypoint acceptance sphere
BASE_RADIUS   = 0.30    # m  tighter acceptance for the landing pad
SETPOINT_HZ   = 20.0    # Hz
PRESTREAM_S   = 6.0     # s  extended: let EKF get optical flow data before arming

MAX_XY_SPEED  = 0.6     # m/s — marker-version baseline (100% coverage)
MAX_Z_SPEED   = 0.5     # m/s
WP_TIMEOUT    = 30.0    # s  abandon path waypoint if not reached

UNSTABLE_ANGLE_DEG = 30.0   # lowered from 40.0 — earlier attitude-fault detection (M-01 fix)
STABLE_ANGLE_DEG   = 10.0
STABLE_HOLD_S      = 1.5

RETURN_MAX_SPEED   = 0.3    # m/s cap for RETURN/DESCEND/TAKEOFF nav
RETURN_FINAL_SPEED = 0.2    # m/s cap for the last leg to the landing pad
RETURN_VIA_RADIUS  = 0.6    # m  acceptance radius for via-points
WALL_BRAKE_MIN     = 0.10   # m/s minimum carrot speed inside brake zone

# ── Precision landing (red-pad blob) ──────────────────────────────────────
PRECLAND_ENABLED       = True   # if False, fall back to legacy DESCEND→AUTO.LAND
PRECLAND_SEARCH_ALT    = 2.2    # m  hover here while searching for the pad (wider FOV)
PRECLAND_ALT_ENTER     = 2.4    # m  altitude at which to begin precision homing (must be ≥ SEARCH_ALT)
PRECLAND_ALT_LAND      = 0.55   # m  drop AUTO.LAND below this (within pad tolerance)
PRECLAND_TOL_M         = 0.12   # m  body-frame pad offset to consider centred
PRECLAND_STEP_Z        = 0.08   # m  per-cycle descent step while homing centred
PRECLAND_STEP_XY       = 0.6    # frac of (dx,dy) to apply per cycle (damping)
PRECLAND_TAG_TIMEOUT   = 4.0    # s  fresh-detection window
PRECLAND_SEARCH_TIMEOUT = 20.0  # s  budget to acquire the pad at SEARCH_ALT
PRECLAND_MAX_TIME      = 40.0   # s  total DESCEND budget; fall back to AUTO.LAND

# ── Base station (EKF LOCAL frame — EKF origin = drone arm point) ─────────
# With EKF2_EV_CTRL=0 (pure optical flow), EKF frame starts at (0,0) when
# the drone arms. Gazebo world spawn is (0.80, 0.80), so EKF = world - 0.80.
BASE_X = 0.0
BASE_Y = 0.0

# ── Polygon arena in EKF LOCAL frame ──────────────────────────────────────
# World coords shifted by (-0.80, -0.80) so wall safety distance calculations
# match the EKF position frame. Without this, north/east wall safety was off
# by 0.8m (EKF thought drone was 0.8m further from wall than reality).
ARENA_POLYGON: list[tuple[float, float]] = [
    (-0.80,  -0.80),
    ( 9.868, -0.80),
    ( 9.868,  6.820),
    (-0.80,   6.820),
]

POLY_MARGIN   = 0.50    # m  inner polygon shrink — increased from 0.25 to keep frontier cells further from walls (M-01 fix). Spawn cell still inside thanks to LR/UD symmetry of 0.5 m offset on a 0.75 m grid.

# ── Polygon-based safety thresholds ──────────────────────────────────────
# Distance measured from inner_polygon boundary (positive = inside).
# WALL_STOP_MARGIN = 0.0 — Layer-3 freeze at the inner polygon edge is the
# hard stop; a positive stop margin would prevent movement from the base
# station corner (only ~0.55 m from the inner boundary).
WALL_BRAKE_START = 1.5   # m  brake zone tuned for 0.3 m/s cruise speed
WALL_STOP_MARGIN = 0.0   # m  Layer-3 freeze handles hard stop; keep 0 so spawn can move
WALL_HARD_LIMIT  = 0.5   # m  (unused in poly version, kept for reference)

RETURN_WALL_BRAKE = 1.5  # m  brake start during RETURN/DESCEND
RETURN_WALL_STOP  = 0.25 # m  min wall distance during RETURN

# ── Grid planner ──────────────────────────────────────────────────────────
GRID_RES = 0.75  # m per cell — restored for Test 3

FORBIDDEN = 0
UNKNOWN   = 1
VISITED   = 2


# ─────────────────────────────────────────────────────────────────────────
# Polygon geometry utilities
# ─────────────────────────────────────────────────────────────────────────

def point_in_polygon(p: tuple[float, float], P: list[tuple[float, float]]) -> bool:
    """Ray-casting inclusion test. P must be non-degenerate."""
    x, y = p
    n = len(P)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = P[i]
        xj, yj = P[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _seg_pt_dist(ax: float, ay: float, bx: float, by: float,
                 px: float, py: float) -> float:
    """Closest distance from point to line segment."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)))
    return math.hypot(px - (ax + t*dx), py - (ay + t*dy))


def distance_to_polygon(p: tuple[float, float],
                        P: list[tuple[float, float]]) -> float:
    """
    Signed distance to polygon boundary.
    Positive = inside, negative = outside.
    """
    n = len(P)
    d = min(_seg_pt_dist(P[i][0], P[i][1], P[(i+1)%n][0], P[(i+1)%n][1],
                         p[0], p[1])
            for i in range(n))
    return d if point_in_polygon(p, P) else -d


def inner_polygon(P: list[tuple[float, float]],
                  margin: float) -> list[tuple[float, float]]:
    """
    Shrink a CCW convex polygon inward by `margin` metres.

    Uses vertex-bisector offsets with the correct offset magnitude
    (margin / cos(half-interior-angle)) so each resulting edge is exactly
    `margin` metres inside the original edge.

    Returns [] if margin > polygon inradius (too small to shrink).
    """
    n = len(P)
    result: list[tuple[float, float]] = []

    def _unit(vx: float, vy: float) -> tuple[float, float]:
        l = math.hypot(vx, vy)
        return (vx / l, vy / l) if l > 1e-9 else (0.0, 0.0)

    for i in range(n):
        prev_v = P[(i - 1) % n]
        curr_v = P[i]
        next_v = P[(i + 1) % n]

        # Edge vectors at this vertex
        e1x, e1y = curr_v[0] - prev_v[0], curr_v[1] - prev_v[1]
        e2x, e2y = next_v[0] - curr_v[0], next_v[1] - curr_v[1]

        # Inward normals for CCW polygon: rotate edge 90° CCW = (-ey, ex)
        n1x, n1y = _unit(-e1y, e1x)
        n2x, n2y = _unit(-e2y, e2x)

        # Bisector of the two inward normals
        bx, by = _unit(n1x + n2x, n1y + n2y)

        # Offset magnitude so each edge is exactly `margin` inside original
        dot = n1x * bx + n1y * by          # cos(angle between n1 and bisector)
        if dot < 0.1:                       # very sharp/reflex angle guard
            dot = 0.1
        offset = margin / dot

        result.append((curr_v[0] + bx * offset, curr_v[1] + by * offset))

    # Sanity: centroid of shrunk polygon must still be inside it
    cx = sum(v[0] for v in result) / n
    cy = sum(v[1] for v in result) / n
    return result if point_in_polygon((cx, cy), result) else []


# ─────────────────────────────────────────────────────────────────────────
# 2-D occupancy grid + frontier + A*
# ─────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    """
    2-D grid over the bounding box of a polygon.
    Cell state: FORBIDDEN (outside inner polygon), UNKNOWN, VISITED.
    """

    def __init__(self, polygon: list, margin: float, res: float):
        self._res   = res
        self._inner = inner_polygon(polygon, margin)

        xs = [v[0] for v in polygon]
        ys = [v[1] for v in polygon]
        self._xmin, self._xmax = min(xs), max(xs)
        self._ymin, self._ymax = min(ys), max(ys)

        self._cols = max(1, int(math.ceil((self._xmax - self._xmin) / res)))
        self._rows = max(1, int(math.ceil((self._ymax - self._ymin) / res)))

        # Flat list indexed [row * cols + col] for speed
        self._grid = [FORBIDDEN] * (self._rows * self._cols)

        if self._inner:
            for r in range(self._rows):
                for c in range(self._cols):
                    cx, cy = self._centre(r, c)
                    if point_in_polygon((cx, cy), self._inner):
                        self._grid[r * self._cols + c] = UNKNOWN

    def _centre(self, row: int, col: int) -> tuple[float, float]:
        return (self._xmin + (col + 0.5) * self._res,
                self._ymin + (row + 0.5) * self._res)

    def _to_rc(self, x: float, y: float) -> tuple[int, int]:
        c = max(0, min(self._cols - 1, int((x - self._xmin) / self._res)))
        r = max(0, min(self._rows - 1, int((y - self._ymin) / self._res)))
        return r, c

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        return self._centre(row, col)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return self._to_rc(x, y)

    def mark_visited(self, x: float, y: float):
        r, c = self._to_rc(x, y)
        if self._grid[r * self._cols + c] != FORBIDDEN:
            self._grid[r * self._cols + c] = VISITED

    def state(self, row: int, col: int) -> int:
        return self._grid[row * self._cols + col]

    def frontiers(self) -> list[tuple[int, int]]:
        """UNKNOWN cells with ≥ 1 VISITED 4-connected neighbour."""
        result = []
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r * self._cols + c] != UNKNOWN:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        if self._grid[nr * self._cols + nc] == VISITED:
                            result.append((r, c))
                            break
        return result

    def nearest_frontier(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, col) of closest frontier to world position, or None.
        Tiebreak toward higher rows (north) to keep the drone away from the
        south wall where optical-flow EKF drift causes premature arena exit."""
        frs = self.frontiers()
        if not frs:
            return None
        cr, cc = self._to_rc(x, y)
        return min(frs, key=lambda rc: ((rc[0]-cr)**2 + (rc[1]-cc)**2, -(rc[0])))

    def astar(self, src: tuple[int, int],
              dst: tuple[int, int]) -> list[tuple[int, int]]:
        """
        A* from src to dst.  Traversable: VISITED or UNKNOWN (not FORBIDDEN).
        Returns path from src to dst inclusive, or [] if unreachable.
        """
        sr, sc = src
        gr, gc = dst

        def h(r: int, c: int) -> float:
            return math.hypot(r - gr, c - gc)

        open_heap: list[tuple[float, int, int, int]] = [
            (h(sr, sc), 0, sr, sc)
        ]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score:   dict[tuple[int, int], int] = {(sr, sc): 0}

        while open_heap:
            _, g, r, c = heapq.heappop(open_heap)

            if (r, c) == (gr, gc):
                path: list[tuple[int, int]] = []
                cur = (r, c)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((sr, sc))
                path.reverse()
                return path

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                    continue
                if self._grid[nr * self._cols + nc] == FORBIDDEN:
                    continue
                ng = g + 1
                if ng < g_score.get((nr, nc), 10**9):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))

        return []

    def stats(self) -> dict[str, int]:
        f = u = v = 0
        for s in self._grid:
            if s == FORBIDDEN: f += 1
            elif s == UNKNOWN:  u += 1
            else:               v += 1
        return {'forbidden': f, 'unknown': u, 'visited': v,
                'total': f + u + v}


# ─────────────────────────────────────────────────────────────────────────
# Mission node
# ─────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ExplorationMissionPoly(Node):

    def __init__(self):
        super().__init__('exploration_mission_poly')

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
        # Precision-landing tag offset (body-frame x,y; z = altitude)
        self.create_subscription(PointStamped, '/landing_pad/offset',
                                 self._tag_cb, qos, callback_group=cbg)
        self._tag_dx = 0.0
        self._tag_dy = 0.0
        self._tag_h  = 0.0
        self._tag_fresh = False
        self._tag_last_stamp_s = 0.0
        self._precland_t0 = 0.0

        self._sp_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', qos)
        self._arm_cli  = self.create_client(CommandBool, '/mavros/cmd/arming',
                                            callback_group=cbg)
        self._mode_cli = self.create_client(SetMode, '/mavros/set_mode',
                                            callback_group=cbg)

        # Polygon + grid
        self._outer_poly = ARENA_POLYGON
        self._inner_poly = inner_polygon(ARENA_POLYGON, POLY_MARGIN)
        self._grid = OccupancyGrid(ARENA_POLYGON, POLY_MARGIN, GRID_RES)

        # State machine
        self._mav_state = State()
        self._pose      = PoseStamped()
        self._phase     = 'IDLE'
        self._t_phase   = time.time()

        # Carrot / goal — start at spawn position (inside inner polygon)
        self._goal: tuple[float, float, float] = (BASE_X, BASE_Y, FLIGHT_ALT)
        self._sp   = self._make_sp(BASE_X, BASE_Y, 0.3)
        self._wp_deadline = float('inf')

        # OFFBOARD bootstrap
        self._prestream_done      = False
        self._last_offboard_req_t = 0.0
        self._brake_logged        = False

        # Stability recovery
        self._stabilize_return_to: str | None = None
        self._stable_since:        float | None = None
        self._last_safe_pos: tuple[float, float, float] = (0.0, 0.0, FLIGHT_ALT)

        # Planned path (world waypoints)
        self._path:     list[tuple[float, float, float]] = []
        self._path_idx: int = 0

        # Return path
        self._return_wps:    list[tuple[float, float, float]] = []
        self._return_wp_idx: int = 0

        gs = self._grid.stats()
        self.get_logger().info(
            f'PolyMission ready  grid={self._grid._rows}×{self._grid._cols} '
            f'res={GRID_RES}m  unknown={gs["unknown"]}  forbidden={gs["forbidden"]}')
        self.get_logger().info(
            f'Inner polygon ({len(self._inner_poly)} verts): '
            + '  '.join(f'({x:.2f},{y:.2f})' for x, y in self._inner_poly))

        self._timer = self.create_timer(1.0 / SETPOINT_HZ, self._loop,
                                        callback_group=cbg)

    # ── ROS callbacks ──────────────────────────────────────────────────────
    def _state_cb(self, msg: State):
        self._mav_state = msg

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg

    def _tag_cb(self, msg: PointStamped):
        # detector publishes header.stamp.sec=0 when the detection is stale
        self._tag_dx = msg.point.x
        self._tag_dy = msg.point.y
        self._tag_h  = msg.point.z
        self._tag_fresh = (msg.header.stamp.sec != 0
                           or msg.header.stamp.nanosec != 0)
        if self._tag_fresh:
            self._tag_last_stamp_s = time.time()

    def _yaw(self) -> float:
        q = self._pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ── Low-level helpers ──────────────────────────────────────────────────
    def _make_sp(self, x: float, y: float, z: float) -> PoseStamped:
        sp = PoseStamped()
        sp.header.frame_id = 'map'
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z
        sp.pose.orientation.w = 1.0
        return sp

    def _xyz(self) -> tuple[float, float, float]:
        p = self._pose.pose.position
        return p.x, p.y, p.z

    def _dist_xyz(self, tx: float, ty: float, tz: float) -> float:
        px, py, pz = self._xyz()
        return math.sqrt((px-tx)**2 + (py-ty)**2 + (pz-tz)**2)

    def _dist_goal(self) -> float:
        return self._dist_xyz(*self._goal)

    def _wait_future(self, fut, timeout: float = 3.0) -> bool:
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

    def _roll_pitch_deg(self) -> tuple[float, float]:
        q = self._pose.pose.orientation
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            return 0.0, 0.0
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll  = math.degrees(math.atan2(sinr, cosr))
        sinp  = _clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
        pitch = math.degrees(math.asin(sinp))
        return abs(roll), abs(pitch)

    # ── Polygon speed limits ───────────────────────────────────────────────
    def _boundary_dist(self, x: float, y: float) -> float:
        """Signed dist to inner_polygon boundary. Positive = inside."""
        if not self._inner_poly:
            return 0.0
        return distance_to_polygon((x, y), self._inner_poly)

    def _speed_limit_explore(self, x: float, y: float) -> float:
        """Layer-2 brake zone for EXPLORE."""
        d = self._boundary_dist(x, y)
        if d <= WALL_STOP_MARGIN:
            return 0.0
        if d <= WALL_BRAKE_START:
            t = (d - WALL_STOP_MARGIN) / (WALL_BRAKE_START - WALL_STOP_MARGIN)
            return WALL_BRAKE_MIN + t * (MAX_XY_SPEED - WALL_BRAKE_MIN)
        return MAX_XY_SPEED

    def _speed_limit_return(self, x: float, y: float) -> float:
        """Softer speed limit for RETURN / DESCEND (base station is near wall)."""
        d = self._boundary_dist(x, y)
        if d <= RETURN_WALL_STOP:
            return 0.0
        if d <= RETURN_WALL_BRAKE:
            t = (d - RETURN_WALL_STOP) / (RETURN_WALL_BRAKE - RETURN_WALL_STOP)
            return WALL_BRAKE_MIN + t * (RETURN_MAX_SPEED - WALL_BRAKE_MIN)
        return RETURN_MAX_SPEED

    # ── Carrot-chase setpoint (runs every tick) ────────────────────────────
    def _step_sp_toward_goal(self):
        gx, gy, gz = self._goal
        dt = 1.0 / SETPOINT_HZ
        px = self._sp.pose.position.x
        py = self._sp.pose.position.y
        pz = self._sp.pose.position.z

        dx, dy   = gx - px, gy - py
        dist_xy  = math.hypot(dx, dy)

        if dist_xy > 1e-3:
            dx_n, dy_n = dx / dist_xy, dy / dist_xy

            if self._phase == 'EXPLORE':
                spd = self._speed_limit_explore(px, py)
                if spd < MAX_XY_SPEED * 0.95:
                    if not self._brake_logged:
                        d = self._boundary_dist(px, py)
                        self.get_logger().warn(
                            f'Poly brake: {spd:.2f} m/s  '
                            f'dist_to_inner_boundary={d:.2f} m')
                        self._brake_logged = True
                else:
                    self._brake_logged = False

                step = min(dist_xy, spd * dt)
                nx, ny = px + dx_n * step, py + dy_n * step

                # Layer 3: freeze carrot only when it is currently inside the
                # inner polygon and the next step would exit it. If the carrot
                # is already outside (e.g. teleport / edge case) allow movement
                # so it can recover — freezing an already-outside carrot causes
                # a permanent deadlock at that position.
                if self._inner_poly and \
                        not point_in_polygon((nx, ny), self._inner_poly) and \
                        point_in_polygon((px, py), self._inner_poly):
                    nx, ny = px, py
                    self.get_logger().error(
                        f'Layer-3 freeze: carrot ({nx:.2f},{ny:.2f}) '
                        f'would exit inner polygon')

            else:
                # RETURN / DESCEND / TAKEOFF — softer limits; base station is
                # 0.80 m from wall so RETURN_WALL_STOP=0.25 m keeps it reachable.
                spd = self._speed_limit_return(px, py)
                if (self._phase == 'RETURN' and self._return_wps and
                        self._return_wp_idx == len(self._return_wps) - 1):
                    spd = min(spd, RETURN_FINAL_SPEED)
                step = min(dist_xy, spd * dt)
                nx, ny = px + dx_n * step, py + dy_n * step

            px, py = nx, ny
        else:
            px, py = gx, gy

        dz = gz - pz
        if abs(dz) > 1e-3:
            pz += math.copysign(min(abs(dz), MAX_Z_SPEED * dt), dz)
        else:
            pz = gz

        self._sp = self._make_sp(px, py, pz)

    # ── Frontier planner ───────────────────────────────────────────────────
    def _plan_to_next_frontier(self, px: float, py: float) -> bool:
        """
        Find nearest frontier, A* path, convert to world waypoints.
        Returns True if a valid path was planned, False if arena is fully explored.
        """
        frontier = self._grid.nearest_frontier(px, py)
        if frontier is None:
            return False

        # Skip frontier cells whose centre is within WP_RADIUS of the inner
        # polygon boundary — the carrot can't reach them without Layer-3
        # freezing, so mark them VISITED immediately and try the next one.
        fr, fc = frontier
        fx, fy = self._grid.cell_centre(fr, fc)
        if self._inner_poly and \
                distance_to_polygon((fx, fy), self._inner_poly) < WP_RADIUS:
            self._grid._grid[fr * self._grid._cols + fc] = VISITED
            self.get_logger().debug(
                f'Skipping boundary-adjacent frontier ({fr},{fc}) '
                f'({fx:.2f},{fy:.2f}) — too close to inner polygon edge')
            return True   # retry next tick with a different frontier

        start_rc = self._grid.world_to_cell(px, py)
        path_rc  = self._grid.astar(start_rc, frontier)

        if not path_rc:
            # Isolated frontier — mark it visited so we skip it next tick
            self._grid._grid[frontier[0] * self._grid._cols + frontier[1]] = VISITED
            self.get_logger().warn(
                f'A* cannot reach frontier {frontier} — skipping')
            return True   # retry with a different frontier next EXPLORE tick

        # Convert to world waypoints, gate each against inner polygon
        self._path     = []
        self._path_idx = 0
        for r, c in path_rc[1:]:          # skip current cell
            cx, cy = self._grid.cell_centre(r, c)
            if self._inner_poly and point_in_polygon((cx, cy), self._inner_poly):
                self._path.append((cx, cy, FLIGHT_ALT))
            else:
                self.get_logger().warn(
                    f'Path cell ({r},{c}) centre ({cx:.2f},{cy:.2f}) '
                    f'outside inner poly — truncating path')
                break

        if not self._path:
            self._grid._grid[frontier[0] * self._grid._cols + frontier[1]] = VISITED
            return True

        gx, gy, gz = self._path[0]
        self._goal        = (gx, gy, gz)
        self._wp_deadline = time.time() + WP_TIMEOUT
        gs = self._grid.stats()
        self.get_logger().info(
            f'Frontier ({frontier[0]},{frontier[1]})  '
            f'path={len(self._path)} WPs  '
            f'unknown={gs["unknown"]}  visited={gs["visited"]}')
        return True

    def _advance_path(self):
        """Move to next path waypoint, or replan when path is exhausted."""
        self._path_idx += 1
        if self._path_idx < len(self._path):
            gx, gy, gz = self._path[self._path_idx]
            self._goal        = (gx, gy, gz)
            self._wp_deadline = time.time() + WP_TIMEOUT
        else:
            px, py, _ = self._xyz()
            if not self._plan_to_next_frontier(px, py):
                self.get_logger().info('Arena fully explored. Returning to base.')
                self._start_return()

    def _start_return(self):
        """Build staged return path and transition to RETURN."""
        px, py, _ = self._xyz()
        self._return_wps    = []
        self._return_wp_idx = 0
        if py > 2.7:   # EKF frame: world 3.5 - 0.8 = 2.7
            self._return_wps.append((4.2, 2.7, FLIGHT_ALT))   # arena centre via-point (EKF frame)
        self._return_wps.append((BASE_X, BASE_Y, FLIGHT_ALT))
        rx, ry, rz = self._return_wps[0]
        self._goal  = (rx, ry, rz)
        self._phase = 'RETURN'
        self.get_logger().info(
            'Return path: '
            + ' → '.join(f'({w[0]:.1f},{w[1]:.1f})' for w in self._return_wps))

    # ── Main control loop (20 Hz) ──────────────────────────────────────────
    def _loop(self):
        self._step_sp_toward_goal()
        self._sp.header.stamp = self.get_clock().now().to_msg()
        self._sp_pub.publish(self._sp)

        # ── Stability monitor ──────────────────────────────────────────────
        if self._phase not in ('IDLE', 'PRESTREAM', 'LAND', 'DONE', 'STABILIZE'):
            px, py, pz = self._xyz()
            roll, pitch = self._roll_pitch_deg()
            near_ground = (self._phase == 'TAKEOFF' and pz < 1.5)
            if not near_ground and \
                    (roll > UNSTABLE_ANGLE_DEG or pitch > UNSTABLE_ANGLE_DEG):
                sx, sy, sz = self._last_safe_pos
                self.get_logger().error(
                    f'UNSTABLE: roll={roll:.1f}° pitch={pitch:.1f}° at '
                    f'({px:.2f},{py:.2f},{pz:.2f}) — recovering to '
                    f'({sx:.2f},{sy:.2f},{sz:.2f})')
                self._stabilize_return_to = self._phase
                self._goal         = (sx, sy, sz)
                self._stable_since = None
                self._phase        = 'STABILIZE'
                return
            else:
                if point_in_polygon((px, py), self._outer_poly):
                    self._last_safe_pos = (px, py, max(pz, FLIGHT_ALT))
                else:
                    # EKF drifted outside outer polygon — land rather than
                    # chase phantom corrections
                    self.get_logger().error(
                        f'EKF drift: ({px:.2f},{py:.2f}) outside arena — AUTO.LAND')
                    self._set_mode('AUTO.LAND')
                    self._phase = 'LAND'
                    return

            if self._phase == 'EXPLORE':
                self._grid.mark_visited(px, py)

        # ── OFFBOARD watchdog ──────────────────────────────────────────────
        if (self._phase not in ('IDLE', 'PRESTREAM', 'LAND', 'DONE') and
                self._mav_state.armed and
                self._mav_state.mode != 'OFFBOARD'):
            if self._mav_state.mode.startswith('AUTO'):
                # PX4 entered a failsafe (AUTO.LAND, AUTO.RTL, etc.) — yield to it
                if self._phase != 'LAND':
                    self.get_logger().warn(
                        f'PX4 entered {self._mav_state.mode} — yielding, setting phase=LAND')
                    self._phase = 'LAND'
            elif time.time() - self._last_offboard_req_t > 1.0:
                self._last_offboard_req_t = time.time()
                req = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                self._mode_cli.call_async(req)
                self.get_logger().warn(
                    f'OFFBOARD lost ({self._mav_state.mode}) — re-requesting…')

        # ── IDLE ──────────────────────────────────────────────────────────
        if self._phase == 'IDLE':
            if not self._mav_state.connected:
                return
            self.get_logger().info('MAVROS connected. Pre-streaming setpoints…')
            self._phase   = 'PRESTREAM'
            self._t_phase = time.time()

        # ── PRESTREAM ─────────────────────────────────────────────────────
        elif self._phase == 'PRESTREAM':
            if time.time() - self._t_phase < PRESTREAM_S:
                return
            if self._prestream_done:
                return
            # Block arming until EKF has published a valid orientation
            q = self._pose.pose.orientation
            if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
                self.get_logger().warn('EKF pose not ready — delaying arm...')
                return
            self._prestream_done = True
            if not self._set_mode('OFFBOARD'):
                self._t_phase = time.time()
                return
            if not self._arm(True):
                self._prestream_done = False
                self._t_phase = time.time()
                return
            self.get_logger().info(f'Armed. Climbing to {FLIGHT_ALT} m…')
            self._goal  = (0.0, 0.0, FLIGHT_ALT)
            self._phase = 'TAKEOFF'

        # ── TAKEOFF ───────────────────────────────────────────────────────
        elif self._phase == 'TAKEOFF':
            px, py, z = self._xyz()
            self._goal = (px, py, FLIGHT_ALT)   # hold XY during climb
            if z >= FLIGHT_ALT - 0.30:
                self.get_logger().info(
                    f'At {z:.2f} m. Starting frontier exploration.')
                self._grid.mark_visited(px, py)  # spawn cell → VISITED
                if not self._plan_to_next_frontier(px, py):
                    self.get_logger().warn('No frontiers at takeoff — landing.')
                    self._start_return()
                else:
                    self._phase = 'EXPLORE'
            elif time.time() - self._t_phase > 20.0 and z < 0.5:
                self.get_logger().error(
                    f'TAKEOFF abort: z={z:.2f} m after 20 s.')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

        # ── EXPLORE ───────────────────────────────────────────────────────
        elif self._phase == 'EXPLORE':
            timed_out = time.time() > self._wp_deadline
            if self._dist_goal() < WP_RADIUS or timed_out:
                if timed_out:
                    self.get_logger().warn(
                        f'Path WP {self._path_idx} timed out — advancing.')
                    # Mark the unreachable goal cell as visited so the planner
                    # won't target it again (handles corner cells at inner polygon
                    # boundary that the drone can't physically reach).
                    gx, gy, _ = self._goal
                    self._grid.mark_visited(gx, gy)
                # Debounce: prevents 20 Hz hot-loop when carrot hasn't moved yet
                if time.time() - getattr(self, '_last_plan_t', 0) > 0.3:
                    self._last_plan_t = time.time()
                    self._advance_path()

        # ── RETURN ────────────────────────────────────────────────────────
        elif self._phase == 'RETURN':
            is_last = (self._return_wp_idx == len(self._return_wps) - 1)
            radius  = BASE_RADIUS if is_last else RETURN_VIA_RADIUS
            if self._dist_goal() < radius:
                self._return_wp_idx += 1
                if self._return_wp_idx >= len(self._return_wps):
                    # Climb to wide-FOV search altitude over EKF base; the
                    # PAD_SEARCH phase will hover here until the red pad is
                    # acquired in-frame, then transition to DESCEND.
                    self.get_logger().info(
                        f'Above base. Climbing to {PRECLAND_SEARCH_ALT} m to '
                        f'acquire pad.')
                    self._goal  = (BASE_X, BASE_Y, PRECLAND_SEARCH_ALT)
                    self._phase = 'PAD_SEARCH'
                    self._precland_t0 = time.time()
                else:
                    rx, ry, rz = self._return_wps[self._return_wp_idx]
                    self.get_logger().info(
                        f'Return leg {self._return_wp_idx}: '
                        f'→ ({rx:.1f},{ry:.1f})')
                    self._goal = (rx, ry, rz)

        # ── PAD_SEARCH — hover at wide-FOV altitude until pad is in view ─
        elif self._phase == 'PAD_SEARCH':
            px, py, z = self._xyz()
            elapsed = time.time() - self._precland_t0
            tag_age = (time.time() - self._tag_last_stamp_s
                       if self._tag_last_stamp_s > 0.0 else 9999.0)

            # Fresh detection → home one step toward pad (still at search alt),
            # then transition to DESCEND on the next cycle so the homing loop
            # takes over from this position.
            if self._tag_fresh and tag_age <= PRECLAND_TAG_TIMEOUT:
                yaw = self._yaw()
                cy, sy = math.cos(yaw), math.sin(yaw)
                dwx = self._tag_dx * cy - self._tag_dy * sy
                dwy = self._tag_dx * sy + self._tag_dy * cy
                err_xy = math.hypot(self._tag_dx, self._tag_dy)
                self.get_logger().info(
                    f'PAD_SEARCH: pad acquired (err={err_xy*100:.0f}cm) '
                    f'→ DESCEND')
                # Set initial goal at search altitude over the pad
                self._goal  = (px + _clamp(0.7 * dwx, -0.6, 0.6),
                               py + _clamp(0.7 * dwy, -0.6, 0.6),
                               PRECLAND_SEARCH_ALT)
                self._phase = 'DESCEND'
                self._precland_t0 = time.time()  # reset DESCEND budget

            # Timeout — give up search, fall back to plain AUTO.LAND
            elif elapsed > PRECLAND_SEARCH_TIMEOUT:
                self.get_logger().warn(
                    f'PAD_SEARCH timeout {elapsed:.1f}s — fallback AUTO.LAND '
                    f'at ({px:.2f},{py:.2f})')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

        # ── DESCEND (precision landing on red pad) ───────────────────────
        elif self._phase == 'DESCEND':
            px, py, z = self._xyz()
            elapsed = time.time() - self._precland_t0
            tag_age = (time.time() - self._tag_last_stamp_s
                       if self._tag_last_stamp_s > 0.0 else 9999.0)

            # Budget exceeded — give up and land where we are
            if elapsed > PRECLAND_MAX_TIME:
                self.get_logger().warn(
                    f'PRECLAND timeout {elapsed:.1f}s — AUTO.LAND at '
                    f'({px:.2f},{py:.2f},{z:.2f})')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

            # Precision homing only if enabled, tag fresh, and altitude allows
            elif (PRECLAND_ENABLED and self._tag_fresh
                  and tag_age <= PRECLAND_TAG_TIMEOUT
                  and z <= PRECLAND_ALT_ENTER):
                dx_b, dy_b = self._tag_dx, self._tag_dy
                # body (FLU: x forward, y left) → world ENU using current yaw
                yaw = self._yaw()
                cy, sy = math.cos(yaw), math.sin(yaw)
                dwx = dx_b * cy - dy_b * sy
                dwy = dx_b * sy + dy_b * cy
                # Apply with damping; clamp single-step correction
                dwx = _clamp(PRECLAND_STEP_XY * dwx, -0.4, 0.4)
                dwy = _clamp(PRECLAND_STEP_XY * dwy, -0.4, 0.4)
                # Vertical step: keep descending each cycle
                err_xy = math.hypot(dx_b, dy_b)
                # If not centred, slow the descent; if centred, descend normally
                step_z = (PRECLAND_STEP_Z
                          if err_xy < PRECLAND_TOL_M
                          else PRECLAND_STEP_Z * 0.4)
                gx = px + dwx
                gy = py + dwy
                gz = max(PRECLAND_ALT_LAND - 0.05, z - step_z)
                self._goal = (gx, gy, gz)

                # Trigger AUTO.LAND if low enough and well-centred
                if z <= PRECLAND_ALT_LAND and err_xy < PRECLAND_TOL_M * 2.0:
                    self.get_logger().info(
                        f'PRECLAND OK: err={err_xy*100:.1f}cm z={z:.2f}m → AUTO.LAND')
                    self._set_mode('AUTO.LAND')
                    self._phase = 'LAND'

            # Legacy fallback: no fresh tag — proceed to plain DESCEND→LAND
            elif z <= DESCENT_ALT + 0.20:
                if tag_age > PRECLAND_TAG_TIMEOUT:
                    self.get_logger().warn(
                        f'PRECLAND no tag for {tag_age:.1f}s — fallback AUTO.LAND')
                else:
                    self.get_logger().info('Triggering AUTO.LAND.')
                self._set_mode('AUTO.LAND')
                self._phase = 'LAND'

        # ── STABILIZE ─────────────────────────────────────────────────────
        elif self._phase == 'STABILIZE':
            roll, pitch = self._roll_pitch_deg()
            if roll < STABLE_ANGLE_DEG and pitch < STABLE_ANGLE_DEG:
                if self._stable_since is None:
                    self._stable_since = time.time()
                    self.get_logger().info(
                        f'Attitude normalising — holding {STABLE_HOLD_S} s…')
                elif time.time() - self._stable_since >= STABLE_HOLD_S:
                    resume = self._stabilize_return_to
                    self.get_logger().info(f'Stable — resuming {resume}')
                    self._phase = resume
                    if resume == 'EXPLORE':
                        px, py, _ = self._xyz()
                        self._plan_to_next_frontier(px, py)
                    elif resume == 'RETURN':
                        if (self._return_wps and
                                self._return_wp_idx < len(self._return_wps)):
                            self._goal = self._return_wps[self._return_wp_idx]
                        else:
                            self._goal = (BASE_X, BASE_Y, FLIGHT_ALT)
                    elif resume == 'DESCEND':
                        self._goal = (BASE_X, BASE_Y, DESCENT_ALT)
                    elif resume == 'PAD_SEARCH':
                        self._goal = (BASE_X, BASE_Y, PRECLAND_SEARCH_ALT)
            else:
                self._stable_since = None

        # ── LAND ──────────────────────────────────────────────────────────
        elif self._phase == 'LAND':
            if not self._mav_state.armed:
                gs = self._grid.stats()
                self.get_logger().info(
                    f'Landed. Mission complete.  '
                    f'visited={gs["visited"]}  unknown={gs["unknown"]}  '
                    f'forbidden={gs["forbidden"]}')
                self._phase = 'DONE'
                raise SystemExit(0)


def main():
    rclpy.init()
    node = ExplorationMissionPoly()
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
