#!/usr/bin/python3
"""
GPS-denied Offboard takeoff-hover-land mission.
ROS 2 Jazzy + MAVROS + PX4 SITL + Gazebo Harmonic

Architecture:
  - executor (main thread) : spins node — handles subscriptions, timer setpoint publishing
  - control_thread         : calls services (set_mode, arm) WITHOUT being inside a callback
                             polls future.done() while executor runs freely in main thread
"""

import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State


TARGET_ALT = 3.0    # metres ENU
HOVER_SECS = 30.0
STREAM_HZ  = 20

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class OffboardMission(Node):
    def __init__(self):
        super().__init__('offboard_mission')

        self.state   = State()
        self.local_z = 0.0
        self._sp_z   = TARGET_ALT   # controlled by control thread
        self._first_pose_logged = False

        # Publishers
        self._sp_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        # Subscribers — BEST_EFFORT to match MAVROS publisher QoS
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self._pose_cb, SENSOR_QOS)

        # Service clients
        self._mode_cli = self.create_client(SetMode,     '/mavros/set_mode')
        self._arm_cli  = self.create_client(CommandBool, '/mavros/cmd/arming')

        # Setpoint message (z updated by control thread)
        self._sp = PoseStamped()
        self._sp.header.frame_id    = 'map'
        self._sp.pose.position.z    = self._sp_z
        self._sp.pose.orientation.w = 1.0

        # Timer publishes setpoints at STREAM_HZ (stays in executor, no service calls)
        self.create_timer(1.0 / STREAM_HZ, self._publish_sp)

        # Control thread runs the state machine — calls services outside executor
        self._ctrl = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl.start()

        self.get_logger().info('Mission node started — control thread running.')

    # ------------------------------------------------------------------ #
    # Executor callbacks (no service calls here)                           #
    # ------------------------------------------------------------------ #

    def _state_cb(self, msg: State):
        self.state = msg

    def _pose_cb(self, msg: PoseStamped):
        self.local_z = msg.pose.position.z
        if not self._first_pose_logged:
            self._first_pose_logged = True
            self.get_logger().info(
                f'[SENSOR OK] EKF local pose stream active  z={self.local_z:.3f} m')

    def _publish_sp(self):
        self._sp.header.stamp        = self.get_clock().now().to_msg()
        self._sp.pose.position.z     = self._sp_z
        self._sp_pub.publish(self._sp)

    # ------------------------------------------------------------------ #
    # Service helper — safe from non-callback thread                       #
    # Executor (main thread) keeps spinning and resolves futures freely.   #
    # ------------------------------------------------------------------ #

    def _call(self, client, request, timeout=8.0):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return None
        future   = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().warn(f'{client.srv_name} timed out')
                return None
            time.sleep(0.05)          # yield CPU — executor resolves future
        return future.result()

    def _set_mode(self, mode: str) -> bool:
        req = SetMode.Request()
        req.custom_mode = mode
        res = self._call(self._mode_cli, req)
        ok  = res is not None and res.mode_sent
        self.get_logger().info(f'set_mode({mode}) → {"OK" if ok else "FAILED"}')
        return ok

    def _arm(self, value: bool) -> bool:
        req = CommandBool.Request()
        req.value = value
        res = self._call(self._arm_cli, req)
        ok  = res is not None and res.success
        self.get_logger().info(f'arm({value}) → {"OK" if ok else "FAILED"}')
        return ok

    # ------------------------------------------------------------------ #
    # Control loop (runs in control thread, NOT in executor)               #
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        log = self.get_logger()

        # 1. Wait for MAVROS connection
        log.info('Waiting for MAVROS connection…')
        _t_conn      = time.time()
        _last_w_log  = time.time()
        while not self.state.connected:
            time.sleep(0.1)
            if time.time() - _last_w_log >= 3.0:
                _last_w_log = time.time()
                log.info(
                    f'  ...still waiting for MAVROS '
                    f'({time.time()-_t_conn:.0f}s)  '
                    f'armed={self.state.armed}  mode={self.state.mode}')
        log.info(
            f'[CONNECTED] MAVROS link established  '
            f'({time.time()-_t_conn:.1f}s)')

        # 2. Pre-stream 2 s so PX4 sees a continuous setpoint before mode switch
        log.info('Pre-streaming setpoints for 2 s…')
        time.sleep(2.0)

        # 3. Switch to OFFBOARD
        while True:
            if self.state.mode == 'OFFBOARD':
                log.info('Already in OFFBOARD mode.')
                break
            if self._set_mode('OFFBOARD'):
                break
            log.warn('set_mode OFFBOARD failed, retrying in 1 s…')
            time.sleep(1.0)

        # 4. Arm
        time.sleep(0.5)
        while not self.state.armed:
            if self._arm(True):
                break
            log.warn('Arm failed, retrying in 1 s…')
            time.sleep(1.0)

        log.info('=' * 48)
        log.info('  *** DRONE ARMED — ASCENDING ***')
        log.info(f'  Mode: {self.state.mode}  |  Target: {TARGET_ALT} m')
        log.info('=' * 48)

        # 5. Climb
        _last_alt_log = -1.0
        while self.local_z < TARGET_ALT - 0.3:
            time.sleep(0.1)
            if self.local_z - _last_alt_log >= 0.4:
                _last_alt_log = self.local_z
                log.info(
                    f'  [CLIMB] alt={self.local_z:.2f}m → {TARGET_ALT}m  '
                    f'mode={self.state.mode}')
        log.info(
            f'[ALTITUDE REACHED] {self.local_z:.2f} m. '
            f'Hovering {HOVER_SECS} s…')

        # 6. Hover — with countdown every 5 s
        _hover_t0 = time.time()
        _next_log  = 0.0
        while time.time() - _hover_t0 < HOVER_SECS:
            elapsed = time.time() - _hover_t0
            if elapsed >= _next_log:
                _next_log = elapsed + 5.0
                log.info(
                    f'  [HOVER] {elapsed:.0f}s / {HOVER_SECS:.0f}s  '
                    f'z={self.local_z:.2f}m  mode={self.state.mode}')
            time.sleep(0.2)
        log.info('Hover complete. Switching to AUTO.LAND…')

        # 7. Hand landing to PX4 — AUTO.LAND owns descent, land-detect, auto-disarm
        while True:
            if self._set_mode('AUTO.LAND'):
                break
            log.warn('AUTO.LAND switch failed, retrying in 1 s…')
            time.sleep(1.0)

        # 8. Wait for PX4 land detector to confirm and auto-disarm (up to 60 s)
        log.info('Descending via AUTO.LAND… waiting for auto-disarm.')
        deadline      = time.time() + 60.0
        _last_lnd_log = 0.0
        while self.state.armed and time.time() < deadline:
            time.sleep(0.5)
            if time.time() - _last_lnd_log >= 3.0:
                _last_lnd_log = time.time()
                log.info(
                    f'  [LANDING] z={self.local_z:.2f}m  '
                    f'mode={self.state.mode}  armed={self.state.armed}')
        if not self.state.armed:
            log.info('Landed and disarmed. Mission complete.')
        else:
            log.warn('Timed out waiting for auto-disarm. Mission complete.')


def main():
    rclpy.init()
    node     = OffboardMission()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()           # main thread — resolves service futures freely
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
