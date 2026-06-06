#!/usr/bin/python3
"""
Phase 2B — VIO pose publisher (SITL simulation of a real VIO/SLAM pipeline).

Reads Gazebo ground-truth pose for x500_vision_0 via gz.transport13,
converts to ROS PoseStamped, and publishes to /mavros/vision_pose/pose at 30 Hz.

On real hardware this node is replaced by OpenVINS / VINS-Fusion / ORB-SLAM3.
EKF2_EV_CTRL=15 already set — EKF2 fuses /mavros/vision_pose/pose as its
external vision (EV) source.

Frame convention:
  Gazebo world frame = ENU  (x=East, y=North, z=Up).
  /mavros/vision_pose/pose  expects map frame = ENU.
  No rotation needed — direct coordinate pass-through.

Note (Phase 2B): the gz_x500_vision model also publishes odometry via
  /mavros/odometry/in (automatic gz bridge → odometry plugin).
  Both sources are fused by EKF2 simultaneously in this phase.
  Phase 2C will disable the automatic bridge so VIO is the sole source.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped

import gz.transport13
from gz.msgs10.pose_v_pb2 import Pose_V

VIO_HZ     = 30.0
WORLD_NAME = "irocu_arena"
MODEL_NAME = "x500_flow_0"
GZ_TOPIC   = f"/world/{WORLD_NAME}/pose/info"


class VioPublisher(Node):

    def __init__(self):
        super().__init__('vio_publisher')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', qos)

        self._latest: PoseStamped | None = None
        self._lock   = threading.Lock()
        self._gz_count = 0

        self._gz_node = gz.transport13.Node()
        ok = self._gz_node.subscribe(Pose_V, GZ_TOPIC, self._gz_cb)
        if not ok:
            self.get_logger().error(f'VIO: failed to subscribe to {GZ_TOPIC}')
        else:
            self.get_logger().info(f'VIO: subscribed to {GZ_TOPIC}')

        self.create_timer(1.0 / VIO_HZ, self._publish_cb)

    # ── gz transport callback (runs in gz thread) ────────────────────────────

    def _gz_cb(self, msg: Pose_V) -> None:
        for pose in msg.pose:
            if pose.name != MODEL_NAME:
                continue
            ps = PoseStamped()
            ps.header.frame_id   = 'map'
            ps.pose.position.x   = pose.position.x
            ps.pose.position.y   = pose.position.y
            ps.pose.position.z   = pose.position.z
            ps.pose.orientation.x = pose.orientation.x
            ps.pose.orientation.y = pose.orientation.y
            ps.pose.orientation.z = pose.orientation.z
            ps.pose.orientation.w = pose.orientation.w
            with self._lock:
                self._latest    = ps
                self._gz_count += 1
            if self._gz_count == 1:
                self.get_logger().info(
                    f'[SENSOR OK] Gazebo pose stream active for {MODEL_NAME}  '
                    f'pos=({pose.position.x:.3f},{pose.position.y:.3f},'
                    f'{pose.position.z:.3f})')
            return

    # ── ROS timer callback ───────────────────────────────────────────────────

    def _publish_cb(self) -> None:
        with self._lock:
            ps    = self._latest
            count = self._gz_count

        if ps is None:
            self.get_logger().warn(
                'VIO: no Gazebo pose received yet — is Gazebo running?',
                throttle_duration_sec=5.0)
            return

        # Stamp must be fresh; MAVROS rejects poses with stale timestamps.
        ps.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(ps)

        if count % int(VIO_HZ * 2) == 1:   # log every ~2 s
            p = ps.pose.position
            self.get_logger().info(
                f'VIO pub  pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  '
                f'gz_frames={count}')


def main() -> None:
    rclpy.init()
    node = VioPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
