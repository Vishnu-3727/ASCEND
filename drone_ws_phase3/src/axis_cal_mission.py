#!/usr/bin/env python3
"""
Axis calibration mission for SENS_FLOW_ROT determination.
Sequence: arm → climb to 2m → hold 10s → pure North 5s → hold 5s →
          pure East 5s → hold 5s → pure South 5s → hold 5s → land
Uses OFFBOARD velocity setpoints. Run with:
    MISSION_PY=src/axis_cal_mission.py bash run_sitl.sh
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import time

FLIGHT_ALT  = 2.0     # m
VEL_CMD     = 0.4     # m/s (ENU frame: +x=East, +y=North)
PHASE_DUR   = 5.0     # seconds per velocity phase
HOLD_DUR    = 5.0     # seconds per hold phase
PRESTREAM_S = 6.0     # seconds to pre-stream before arm

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)


class AxisCalMission(Node):
    def __init__(self):
        super().__init__('axis_cal')
        self._state = State()
        self._pose  = PoseStamped()
        self._phase = 'PRESTREAM'
        self._phase_start = time.time()
        self._first_pose_rcvd  = False
        self._first_state_rcvd = False
        self._last_log_t       = 0.0

        self._state_sub = self.create_subscription(State, '/mavros/state',
                                                   self._state_cb, QOS)
        self._pose_sub  = self.create_subscription(PoseStamped,
                                                   '/mavros/local_position/pose',
                                                   self._pose_cb, QOS)
        self._vel_pub   = self.create_publisher(TwistStamped,
                                                '/mavros/setpoint_velocity/cmd_vel',
                                                QOS)
        self._pos_pub   = self.create_publisher(PoseStamped,
                                                '/mavros/setpoint_position/local',
                                                QOS)

        self._arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self._mode_client = self.create_client(SetMode,     '/mavros/set_mode')
        self.create_timer(0.05, self._tick)   # 20 Hz
        self.get_logger().info('AxisCal ready')

    def _state_cb(self, msg):
        prev_armed = self._state.armed
        self._state = msg
        if not self._first_state_rcvd:
            self._first_state_rcvd = True
            self.get_logger().info(
                f'[SENSOR OK] MAVROS state active  '
                f'connected={msg.connected}  mode={msg.mode}')
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
                f'[SENSOR OK] EKF pose active  '
                f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})')

    def _z(self): return self._pose.pose.position.z

    def _arm(self):
        req = CommandBool.Request(); req.value = True
        self._arm_client.call_async(req)

    def _set_mode(self, mode):
        req = SetMode.Request(); req.custom_mode = mode
        self._mode_client.call_async(req)

    def _pub_vel(self, vx=0.0, vy=0.0, vz=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx   # ENU East
        msg.twist.linear.y = vy   # ENU North
        msg.twist.linear.z = vz
        self._vel_pub.publish(msg)

    def _pub_hold(self):
        """Hold current horizontal position at FLIGHT_ALT."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self._pose.pose.position.x
        msg.pose.position.y = self._pose.pose.position.y
        msg.pose.position.z = FLIGHT_ALT
        msg.pose.orientation.w = 1.0
        self._pos_pub.publish(msg)

    def _elapsed(self): return time.time() - self._phase_start
    def _next(self, phase): self._phase = phase; self._phase_start = time.time()

    def _tick(self):
        mode = self._state.mode

        if mode.startswith('AUTO') and self._phase not in ('LAND','DONE'):
            self._next('LAND')

        if self._phase in ('CLIMB', 'NORTH', 'EAST', 'SOUTH',
                           'HOLD1', 'HOLD2', 'HOLD3'):
            _now = time.time()
            if _now - self._last_log_t >= 2.0:
                self._last_log_t = _now
                p = self._pose.pose.position
                self.get_logger().info(
                    f'  [{self._phase}] '
                    f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})m  '
                    f'elapsed={self._elapsed():.1f}s  '
                    f'mode={mode}')

        if self._phase == 'PRESTREAM':
            self._pub_vel()
            if self._elapsed() > PRESTREAM_S:
                self.get_logger().info('Pre-stream done → OFFBOARD + ARM')
                self._set_mode('OFFBOARD')
                time.sleep(0.2)
                self._arm()
                self._next('CLIMB')

        elif self._phase == 'CLIMB':
            self._pub_vel(vz=0.4 if self._z() < FLIGHT_ALT - 0.1 else 0.0)
            if self._z() >= FLIGHT_ALT - 0.1:
                self.get_logger().info(f'At {self._z():.2f}m — HOLD1')
                self._next('HOLD1')

        elif self._phase == 'HOLD1':
            self._pub_hold()
            if self._elapsed() > HOLD_DUR:
                self.get_logger().info('HOLD1 done → NORTH phase')
                self._next('NORTH')

        elif self._phase == 'NORTH':
            self._pub_vel(vx=0.0, vy=VEL_CMD)   # +y = North in ENU
            if self._elapsed() > PHASE_DUR:
                self.get_logger().info('NORTH done → HOLD2')
                self._next('HOLD2')

        elif self._phase == 'HOLD2':
            self._pub_hold()
            if self._elapsed() > HOLD_DUR:
                self.get_logger().info('HOLD2 done → EAST phase')
                self._next('EAST')

        elif self._phase == 'EAST':
            self._pub_vel(vx=VEL_CMD, vy=0.0)   # +x = East in ENU
            if self._elapsed() > PHASE_DUR:
                self.get_logger().info('EAST done → HOLD3')
                self._next('HOLD3')

        elif self._phase == 'HOLD3':
            self._pub_hold()
            if self._elapsed() > HOLD_DUR:
                self.get_logger().info('HOLD3 done → SOUTH phase')
                self._next('SOUTH')

        elif self._phase == 'SOUTH':
            self._pub_vel(vx=0.0, vy=-VEL_CMD)  # -y = South in ENU
            if self._elapsed() > PHASE_DUR:
                self.get_logger().info('SOUTH done → LAND')
                self._next('LAND')

        elif self._phase == 'LAND':
            self._set_mode('AUTO.LAND')
            self._next('DONE')

        elif self._phase == 'DONE':
            pass


def main():
    rclpy.init()
    node = AxisCalMission()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
