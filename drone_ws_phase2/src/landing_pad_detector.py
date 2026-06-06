#!/usr/bin/python3
"""
Phase 2E — Landing-pad detector (red-blob variant).

The landing pad is a 0.50 x 0.50 m bright-red square at world (0.80, 0.80).
We use HSV colour thresholding because basic ambient/diffuse materials are
known to render correctly in this Gazebo+ogre2 build (verified by the floor
markers), whereas PBR/AprilTag textures rendered as solid white in early
tests and failed AprilTag detection.

Subscribes to the downward camera + 2D lidar (altitude) via gz.transport13.
Detects the largest red blob, converts pixel offset → body-frame metric
offset using current altitude, and publishes /landing_pad/offset
(geometry_msgs/PointStamped) at ~10 Hz.

  point.x = dx_body  (forward offset to pad, m,  FLU x+)
  point.y = dy_body  (left   offset to pad, m,  FLU y+)
  point.z = altitude (m)
  header.stamp.sec = 0 if no detection in the last 0.5 s (mission treats
  this as stale and falls back to AUTO.LAND at EKF base).

Image-frame → body-frame mapping (downward camera, mount rotation
(0, +pi/2, 0) and no body yaw correction in this node — the mission
rotates body → world using EKF yaw):
  dx_body =  (cy - v) * h / fy   (forward+ if pad north of nose)
  dy_body = -(u - cx) * h / fx   (left+    if pad left of body)
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
from geometry_msgs.msg import PointStamped

import gz.transport13
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan

# ── Camera intrinsics (must match x500_flow model.sdf) ───────────────────────
IMG_W, IMG_H = 1280, 960
HFOV = 1.74  # rad
FX = (IMG_W / 2.0) / math.tan(HFOV / 2.0)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# ── Gazebo topics — world name from env (Phase 3 arena selector) ────────────
import os as _os
_WORLD = _os.environ.get('PX4_GZ_WORLD',
                        _os.environ.get('ARENA_NAME', 'irocu_arena'))
CAM_TOPIC   = f'/world/{_WORLD}/model/x500_flow_0/link/camera_link/sensor/camera/image'
LIDAR_TOPIC = f'/world/{_WORLD}/model/x500_flow_0/link/lidar_sensor_link/sensor/lidar/scan'

# ── Red HSV thresholds (saturated red wraps around H=0) ─────────────────────
# Gazebo Image is RGB; we convert to HSV via cv2.COLOR_RGB2HSV (H in 0..179)
H_LO_1, H_HI_1 = 0,  10
H_LO_2, H_HI_2 = 170, 180
S_LO, V_LO     = 120, 80
S_HI, V_HI     = 255, 255

MIN_AREA_PX = 200   # ignore tiny red specks (dust, render artefacts)
STALE_SEC   = 0.5


class LandingPadDetector(Node):

    def __init__(self):
        super().__init__('landing_pad_detector')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub = self.create_publisher(PointStamped,
                                          '/landing_pad/offset', qos)

        self._lock = threading.Lock()
        self._altitude = 0.0
        self._last_detect_t = 0.0
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._last_area = 0
        self._frames = 0
        self._detects = 0
        self._first_cam_logged   = False
        self._first_lidar_logged = False

        self._gz = gz.transport13.Node()
        ok1 = self._gz.subscribe(GzImage,     CAM_TOPIC,   self._on_image)
        ok2 = self._gz.subscribe(GzLaserScan, LIDAR_TOPIC, self._on_lidar)
        self.get_logger().info(
            f'LPAD: cam_sub={ok1} lidar_sub={ok2}  fx={FX:.1f} cx={CX:.0f}')
        self.get_logger().info(
            f'LPAD: red-blob detector  pad=0.50x0.50m@({0.80},{0.80})  '
            f'min_area={MIN_AREA_PX}px')

        self.create_timer(0.1, self._publish_cb)
        self.create_timer(1.0, self._log_status)

    # ── lidar → altitude ────────────────────────────────────────────────────
    def _on_lidar(self, msg: GzLaserScan):
        ranges = list(msg.ranges)
        valid = [r for r in ranges if 0.05 < r < 30.0 and not math.isnan(r)]
        if valid:
            with self._lock:
                self._altitude = min(valid)
            if not self._first_lidar_logged:
                self._first_lidar_logged = True
                self.get_logger().info(
                    f'[SENSOR OK] Lidar altitude stream active  '
                    f'alt={self._altitude:.3f} m')

    # ── camera → red-blob detection ─────────────────────────────────────────
    def _on_image(self, msg: GzImage):
        if not self._first_cam_logged:
            self._first_cam_logged = True
            self.get_logger().info(
                f'[SENSOR OK] Camera stream active  {msg.width}x{msg.height} px')
        self._frames += 1
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size != msg.width * msg.height * 3:
            return
        img = arr.reshape(msg.height, msg.width, 3)   # RGB

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv, np.array([H_LO_1, S_LO, V_LO]),
                              np.array([H_HI_1, S_HI, V_HI]))
        m2 = cv2.inRange(hsv, np.array([H_LO_2, S_LO, V_LO]),
                              np.array([H_HI_2, S_HI, V_HI]))
        mask = cv2.bitwise_or(m1, m2)

        # Clean up
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        # Largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
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
            h = self._altitude
        if h < 0.15:
            h = 0.15

        dx_body = (CY - v) * h / FY     # forward+
        dy_body = -(u - CX) * h / FX    # left+

        # Save first detection frame as debug
        if self._detects == 0:
            try:
                annotated = img.copy()
                cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (int(u), int(v)), 8, (0, 255, 0), -1)
                cv2.imwrite('/tmp/lpad_first_detection.png',
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                self.get_logger().info(
                    'LPAD: FIRST DETECTION! Saved /tmp/lpad_first_detection.png')
            except Exception:
                pass

        with self._lock:
            self._last_dx = dx_body
            self._last_dy = dy_body
            self._last_area = int(area)
            self._last_detect_t = time.monotonic()
            self._detects += 1

    # ── periodic publish ────────────────────────────────────────────────────
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
            age  = (time.monotonic() - self._last_detect_t
                    if self._last_detect_t else -1.0)
            alt  = self._altitude
            dx   = self._last_dx
            dy   = self._last_dy
            area = self._last_area
        fresh  = (self._last_detect_t > 0.0 and age <= STALE_SEC)
        status = 'PAD LOCKED' if fresh else 'searching...'
        self.get_logger().info(
            f'LPAD [{status}]  '
            f'frames={self._frames}  detects={self._detects}  '
            f'alt={alt:.2f}m  offset=({dx:+.3f},{dy:+.3f})m  '
            f'area={area}px  age={age:.1f}s')


def main():
    rclpy.init()
    node = LandingPadDetector()
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
