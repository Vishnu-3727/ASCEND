#!/usr/bin/python3
"""
Landing-pad detector — real hardware variant (Pi 5, no Gazebo).

Subscribes to:
  /camera/image_raw   (sensor_msgs/Image)   — downward-facing camera
  /mavros/local_position/pose               — altitude (EKF2 z)

Publishes:
  /landing_pad/offset (geometry_msgs/PointStamped)
    point.x = dx_body  (forward, FLU)
    point.y = dy_body  (left,    FLU)
    point.z = altitude (m)
    header.stamp.sec = 0  when detection is stale (>0.5 s old)

Set camera intrinsics via ROS params or override IMG_W/IMG_H/HFOV below.
Run with:
    ros2 run ascend landing_pad_detector_hw
    # or
    python3 src/landing_pad_detector_hw.py
"""
from __future__ import annotations
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── Camera intrinsics — override to match your real camera ───────────────────
IMG_W, IMG_H = 640, 480
HFOV = 1.2   # rad (~69 deg, typical wide-angle Pi cam); tune to your lens
FX = (IMG_W / 2.0) / math.tan(HFOV / 2.0)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# ── Red HSV thresholds (tune under real lighting) ────────────────────────────
H_LO_1, H_HI_1 = 0,  10
H_LO_2, H_HI_2 = 170, 180
S_LO, V_LO     = 100, 60
S_HI, V_HI     = 255, 255

MIN_AREA_PX = 150
STALE_SEC   = 0.5


class LandingPadDetectorHW(Node):

    def __init__(self):
        super().__init__('landing_pad_detector')
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # state
        self._altitude = 1.0    # fallback 1 m until MAVROS pose arrives
        self._last_detect_t = 0.0
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._last_area = 0
        self._frames = 0
        self._detects = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self._pub = self.create_publisher(PointStamped, '/landing_pad/offset', sensor_qos)

        self.create_subscription(Image, '/camera/image_raw', self._on_image, sensor_qos)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._on_pose, sensor_qos)

        self.create_timer(0.1, self._publish_cb)
        self.create_timer(1.0, self._log_status)

        self.get_logger().info(
            f'Landing pad detector (HW) started  '
            f'fx={FX:.1f}  cx={CX:.0f}  min_area={MIN_AREA_PX}px')

    def _on_pose(self, msg: PoseStamped):
        z = msg.pose.position.z
        if z > 0.05:
            with self._lock:
                self._altitude = z

    def _on_image(self, msg: Image):
        self._frames += 1
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}', throttle_duration_sec=5.0)
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([H_LO_1, S_LO, V_LO]),
                              np.array([H_HI_1, S_HI, V_HI]))
        m2 = cv2.inRange(hsv, np.array([H_LO_2, S_LO, V_LO]),
                              np.array([H_HI_2, S_HI, V_HI]))
        mask = cv2.bitwise_or(m1, m2)

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            return
        M = cv2.moments(c)
        if M['m00'] <= 0:
            return
        u = M['m10'] / M['m00']
        v = M['m01'] / M['m00']

        with self._lock:
            h = max(self._altitude, 0.15)

        dx_body = (CY - v) * h / FY
        dy_body = -(u - CX) * h / FX

        if self._detects == 0:
            try:
                annotated = img.copy()
                cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (int(u), int(v)), 8, (0, 255, 0), -1)
                cv2.imwrite('/tmp/lpad_first_detection.png', annotated)
                self.get_logger().info('LPAD: FIRST DETECTION — saved /tmp/lpad_first_detection.png')
            except Exception:
                pass

        with self._lock:
            self._last_dx = dx_body
            self._last_dy = dy_body
            self._last_area = int(area)
            self._last_detect_t = time.monotonic()
            self._detects += 1

    def _publish_cb(self):
        with self._lock:
            age = time.monotonic() - self._last_detect_t
            dx, dy, h = self._last_dx, self._last_dy, self._altitude

        m = PointStamped()
        m.header.frame_id = 'base_link'
        if self._last_detect_t > 0.0 and age <= STALE_SEC:
            m.header.stamp = self.get_clock().now().to_msg()
        else:
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
        m.point.x = float(dx)
        m.point.y = float(dy)
        m.point.z = float(h)
        self._pub.publish(m)

    def _log_status(self):
        with self._lock:
            age  = (time.monotonic() - self._last_detect_t if self._last_detect_t else -1.0)
            alt  = self._altitude
            dx   = self._last_dx
            dy   = self._last_dy
            area = self._last_area
        fresh  = (self._last_detect_t > 0.0 and age <= STALE_SEC)
        status = 'PAD LOCKED' if fresh else 'searching...'
        self.get_logger().info(
            f'LPAD [{status}]  frames={self._frames}  detects={self._detects}  '
            f'alt={alt:.2f}m  offset=({dx:+.3f},{dy:+.3f})m  area={area}px  age={age:.1f}s')


def main():
    rclpy.init()
    node = LandingPadDetectorHW()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
