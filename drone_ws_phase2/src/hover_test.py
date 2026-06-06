#!/usr/bin/python3
"""
Hover test — validates SENS_FLOW_ROT by holding (0,0,2m) for 60 s.

Pass/fail criteria:
  PASS:  EKF (x,y) stays within ±0.15 m of (0,0) throughout the hold.
  DRIFT: Any sample outside ±0.15 m — logs axis, magnitude, and direction.
  FAIL:  EKF exits ±0.50 m — immediate AUTO.LAND.

Run via:  run_hover_test.sh   (swaps in this script instead of exploration_mission_poly.py)

Sequence:  IDLE → PRESTREAM (6 s) → TAKEOFF → HOLD (60 s) → LAND → report
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

FLIGHT_ALT   = 2.0    # m
HOLD_S       = 60.0   # hold duration
SETPOINT_HZ  = 20.0
PRESTREAM_S  = 6.0
MAX_Z_SPEED  = 0.4    # m/s for climb
LOG_INTERVAL = 3.0    # s between drift log lines

DRIFT_WARN  = 0.15    # m  — log warning
DRIFT_ABORT = 0.50    # m  — immediate land


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HoverTest(Node):

    def __init__(self):
        super().__init__('hover_test')
        cbg = ReentrantCallbackGroup()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(State, '/mavros/state', self._state_cb, qos, callback_group=cbg)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, qos, callback_group=cbg)
        self._sp_pub   = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos)
        self._arm_cli  = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=cbg)
        self._mode_cli = self.create_client(SetMode, '/mavros/set_mode', callback_group=cbg)

        self._mav  = State()
        self._pose = PoseStamped()
        self._phase      = 'IDLE'
        self._t_phase    = time.time()
        self._t_hold     = 0.0
        self._t_last_log = 0.0
        self._prestream_done = False
        self._sp_z = 0.3          # climb setpoint starts near ground
        self._last_offboard_t = 0.0

        # Drift tracking
        self._max_drift   = 0.0
        self._max_drift_x = 0.0
        self._max_drift_y = 0.0
        self._samples     = 0
        self._drift_exceeded  = False
        self._first_pose_rcvd = False

        self.create_timer(1.0 / SETPOINT_HZ, self._loop, callback_group=cbg)
        self.get_logger().info(
            f'HoverTest ready — FLIGHT_ALT={FLIGHT_ALT}m  HOLD={HOLD_S}s  '
            f'WARN@{DRIFT_WARN}m  ABORT@{DRIFT_ABORT}m')

    def _state_cb(self, msg):
        prev_armed = self._mav.armed
        self._mav  = msg
        if msg.armed and not prev_armed:
            self.get_logger().info('=' * 48)
            self.get_logger().info('  *** DRONE ARMED ***')
            self.get_logger().info(f'  Mode: {msg.mode}  |  Target: {FLIGHT_ALT} m')
            self.get_logger().info('=' * 48)

    def _pose_cb(self, msg):
        self._pose = msg
        if not self._first_pose_rcvd:
            self._first_pose_rcvd = True
            p = msg.pose.position
            self.get_logger().info(
                f'[SENSOR OK] EKF pose stream active  '
                f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})')

    def _xyz(self):
        p = self._pose.pose.position
        return p.x, p.y, p.z

    def _sp_msg(self, x, y, z):
        sp = PoseStamped()
        sp.header.frame_id = 'map'
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z
        sp.pose.orientation.w = 1.0
        return sp

    def _wait(self, fut, timeout=3.0):
        dl = time.time() + timeout
        while not fut.done() and time.time() < dl:
            time.sleep(0.05)
        return fut.done()

    def _set_mode(self, mode):
        req = SetMode.Request(); req.custom_mode = mode
        fut = self._mode_cli.call_async(req)
        self._wait(fut)
        ok = fut.done() and fut.result() and fut.result().mode_sent
        self.get_logger().info(f'set_mode({mode}) → {"OK" if ok else "FAIL"}')
        return ok

    def _arm(self, v):
        req = CommandBool.Request(); req.value = v
        fut = self._arm_cli.call_async(req)
        self._wait(fut)
        ok = fut.done() and fut.result() and fut.result().success
        self.get_logger().info(f'arm({v}) → {"OK" if ok else "FAIL"}')
        return ok

    def _loop(self):
        px, py, pz = self._xyz()

        # Publish current setpoint every tick
        self._sp_pub.publish(self._sp_msg(0.0, 0.0, self._sp_z))

        # OFFBOARD watchdog — yield to AUTO.* failsafe
        if self._phase not in ('IDLE', 'PRESTREAM', 'LAND', 'DONE') and self._mav.armed:
            if self._mav.mode.startswith('AUTO'):
                if self._phase != 'LAND':
                    self.get_logger().warn(f'PX4 entered {self._mav.mode} — yielding')
                    self._phase = 'LAND'
                return
            if self._mav.mode != 'OFFBOARD' and time.time() - self._last_offboard_t > 1.0:
                self._last_offboard_t = time.time()
                req = SetMode.Request(); req.custom_mode = 'OFFBOARD'
                self._mode_cli.call_async(req)

        if self._phase == 'IDLE':
            if not self._mav.connected:
                return
            self.get_logger().info('MAVROS connected — pre-streaming…')
            self._phase = 'PRESTREAM'
            self._t_phase = time.time()

        elif self._phase == 'PRESTREAM':
            if time.time() - self._t_phase < PRESTREAM_S:
                return
            if self._prestream_done:
                return
            q = self._pose.pose.orientation
            if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
                self.get_logger().warn('EKF pose not ready — waiting…')
                return
            self._prestream_done = True
            if not self._set_mode('OFFBOARD'):
                self._prestream_done = False; self._t_phase = time.time(); return
            if not self._arm(True):
                self._prestream_done = False; self._t_phase = time.time(); return
            self.get_logger().info(f'Armed — climbing to {FLIGHT_ALT} m…')
            self._sp_z = FLIGHT_ALT
            self._t_phase = time.time()
            self._phase = 'TAKEOFF'

        elif self._phase == 'TAKEOFF':
            if pz >= FLIGHT_ALT - 0.25:
                self.get_logger().info(
                    f'[ALTITUDE REACHED] {pz:.2f} m — '
                    f'starting {HOLD_S:.0f} s hover hold.')
                self._t_hold     = time.time()
                self._t_last_log = time.time()
                self._phase = 'HOLD'
            elif time.time() - self._t_phase > 20.0 and pz < 0.5:
                self.get_logger().error(f'TAKEOFF abort: z={pz:.2f} m after 20 s')
                self._set_mode('AUTO.LAND'); self._phase = 'LAND'
            else:
                _now = time.time()
                if _now - getattr(self, '_last_climb_log', 0) >= 2.0:
                    self._last_climb_log = _now
                    self.get_logger().info(
                        f'  [CLIMB] alt={pz:.2f}m → {FLIGHT_ALT}m  '
                        f'mode={self._mav.mode}')

        elif self._phase == 'HOLD':
            # Drift check
            drift = math.hypot(px, py)
            self._samples += 1
            if drift > self._max_drift:
                self._max_drift   = drift
                self._max_drift_x = px
                self._max_drift_y = py

            if drift > DRIFT_ABORT:
                self.get_logger().error(
                    f'ABORT: EKF drift {drift:.3f} m at ({px:.3f},{py:.3f}) — AUTO.LAND')
                self._drift_exceeded = True
                self._set_mode('AUTO.LAND'); self._phase = 'LAND'; return

            # Periodic log
            now = time.time()
            if now - self._t_last_log >= LOG_INTERVAL:
                self._t_last_log = now
                elapsed = now - self._t_hold
                status = 'OK' if drift <= DRIFT_WARN else 'DRIFT'
                self.get_logger().info(
                    f'[HOLD {elapsed:5.1f}s/{HOLD_S:.0f}s]  '
                    f'pos=({px:.3f},{py:.3f},{pz:.3f})m  '
                    f'drift={drift:.3f}m  mode={self._mav.mode}  [{status}]')
                if drift > DRIFT_WARN:
                    # Identify dominant drift axis and direction
                    ax = 'East' if px > 0 else 'West'
                    ay = 'North' if py > 0 else 'South'
                    self.get_logger().warn(
                        f'Drift warning: x={px:+.3f}m ({ax})  y={py:+.3f}m ({ay})')

            # Hold complete
            if now - self._t_hold >= HOLD_S:
                self._print_result()
                self._set_mode('AUTO.LAND'); self._phase = 'LAND'

        elif self._phase == 'LAND':
            if not self._mav.armed:
                self._phase = 'DONE'
                raise SystemExit(0)

    def _print_result(self):
        passed = self._max_drift <= DRIFT_WARN
        verdict = 'PASS ✓' if passed else 'DRIFT DETECTED ✗'
        self.get_logger().info('')
        self.get_logger().info('══════════════════════════════════════════')
        self.get_logger().info(f'  HOVER TEST RESULT: {verdict}')
        self.get_logger().info(f'  Hold duration    : {HOLD_S:.0f} s')
        self.get_logger().info(f'  Samples          : {self._samples}')
        self.get_logger().info(f'  Max drift        : {self._max_drift:.4f} m')
        self.get_logger().info(f'  Max drift pos    : ({self._max_drift_x:+.4f}, {self._max_drift_y:+.4f})')
        self.get_logger().info(f'  Threshold (PASS) : ±{DRIFT_WARN} m')
        if not passed:
            ax = 'East' if self._max_drift_x > 0 else 'West'
            ay = 'North' if self._max_drift_y > 0 else 'South'
            self.get_logger().warn(
                f'  Dominant drift axis: x={self._max_drift_x:+.4f}m ({ax}), '
                f'y={self._max_drift_y:+.4f}m ({ay})')
            self.get_logger().warn(
                '  If drift is still 90° off: try SENS_FLOW_ROT=2 instead of 6')
        self.get_logger().info('══════════════════════════════════════════')


def main():
    rclpy.init()
    node = HoverTest()
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
