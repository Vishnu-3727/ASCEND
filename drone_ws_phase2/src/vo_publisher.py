#!/usr/bin/python3
"""
Phase 2D — Visual Odometry publisher.

Subscribes to the downward 1280×960 camera via gz.transport13,
runs ORB feature tracking + Essential Matrix decomposition to estimate
the drone's lateral displacement, uses the LW20 rangefinder altitude for
metric scale, and publishes PoseStamped to /mavros/vision_pose/pose.

Replaces vio_publisher.py (Gazebo ground-truth cheat) with real CV-based
pose estimation that would work on hardware with a real downward camera.

Pipeline:
  Gazebo camera (gz.transport) → grayscale → ORB keypoints →
  LK optical flow tracking → Essential Matrix (RANSAC) →
  R, t (unit) → scale via rangefinder altitude →
  accumulated pose → /mavros/vision_pose/pose

Scale recovery (monocular):
  At altitude h, scene depth ≈ h (flat ground).
  scale = prev_height / (2 * tan(HFOV/2) / width * |t_px|)
  Simplified: scale_factor = prev_height (metres) — works because
  Essential Matrix t is normalised; we multiply by actual depth h.
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped

import gz.transport13
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan

# ── Camera intrinsics (mono_cam: 1280×960, horizontal_fov=1.74 rad) ──────────
IMG_W, IMG_H = 1280, 960
HFOV   = 1.74                                    # rad
FX     = (IMG_W / 2) / math.tan(HFOV / 2)       # ≈ 556 px
FY     = FX
CX, CY = IMG_W / 2, IMG_H / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# ── Gazebo topics ─────────────────────────────────────────────────────────────
WORLD_NAME   = "irocu_arena"
MODEL_NAME   = "x500_flow_0"
CAM_TOPIC    = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/link/camera_link/sensor/camera/image"
LIDAR_TOPIC  = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/link/lidar_sensor_link/sensor/lidar/scan"

# ── VO tuning ─────────────────────────────────────────────────────────────────
VO_HZ         = 15.0     # publish rate (process every 2nd camera frame @ 30Hz)
MIN_FEATURES  = 10       # minimum tracked features to trust pose estimate
BASE_X, BASE_Y = 0.0, 0.0     # EKF-local frame: arm point = origin (not Gazebo world frame)
BASE_Z         = 0.20         # initial height
MAX_DRIFT_XY   = 15.0    # m — reset if accumulated pose escapes arena (sanity)


def _build_quat_identity() -> tuple:
    return 0.0, 0.0, 0.0, 1.0  # x y z w


class VOPublisher(Node):

    def __init__(self):
        super().__init__('vo_publisher')

        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self._pub = self.create_publisher(
            PoseStamped, '/mavros/vision_pose/pose', qos_rel)
        self._vel_pub = self.create_publisher(
            TwistStamped, '/mavros/vision_speed/speed_twist', qos_rel)

        # ── State ───────────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._height        = BASE_Z     # metres, from rangefinder
        self._pos_x         = BASE_X    # accumulated x (m, ENU)
        self._pos_y         = BASE_Y    # accumulated y (m, ENU)

        self._prev_gray     = None
        self._prev_kp       = None      # list of cv2.KeyPoint
        self._prev_pts      = None      # (N,1,2) float32

        self._frame_count   = 0
        self._pub_count     = 0
        self._track_fail    = 0
        self._debug_saved   = False   # save first good frame for inspection
        self._just_reset    = False   # skip accumulation on first frame after reset
        self._pause_frames  = 0       # don't publish for N frames after reset
        self._vel_x         = 0.0    # instantaneous ENU velocity (m/s)
        self._vel_y         = 0.0
        self._vel_valid     = False  # only publish EV velocity after first real frame

        # ORB detector (downsampled to 640×480 for speed)
        self._orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

        # ── gz.transport subscription ────────────────────────────────────────
        self._gz_node = gz.transport13.Node()
        ok = self._gz_node.subscribe(GzImage, CAM_TOPIC, self._gz_img_cb)
        if ok:
            self.get_logger().info(f'VO: subscribed to {CAM_TOPIC}')
        else:
            self.get_logger().error(f'VO: failed to subscribe to {CAM_TOPIC}')

        ok2 = self._gz_node.subscribe(GzLaserScan, LIDAR_TOPIC, self._gz_lidar_cb)
        if ok2:
            self.get_logger().info(f'VO: subscribed to lidar {LIDAR_TOPIC}')
        else:
            self.get_logger().error(f'VO: failed to subscribe to lidar {LIDAR_TOPIC}')

        self.create_timer(1.0 / VO_HZ, self._publish_cb)

    # ── gz.transport lidar callback (independent altitude source) ───────────

    def _gz_lidar_cb(self, msg: GzLaserScan):
        if msg.ranges and msg.ranges[0] > 0.1:
            with self._lock:
                self._height = msg.ranges[0]

    # ── gz image callback (gz thread) ────────────────────────────────────────

    def _gz_img_cb(self, msg: GzImage):
        self._frame_count += 1
        if self._frame_count % 2 != 0:   # process at ~15 Hz
            return

        # Convert protobuf image to numpy grayscale
        data = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            # pixel_format_type 3=RGB8, 6=BGR8, 1=L8
            fmt = msg.pixel_format_type
            if fmt == 1:    # L8 grayscale
                gray = data.reshape((msg.height, msg.width))
            elif fmt in (3, 6):
                channels = 3
                img = data.reshape((msg.height, msg.width, channels))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if fmt == 3
                                    else cv2.COLOR_BGR2GRAY)
            else:
                # Try RGB8 as default
                img = data.reshape((msg.height, msg.width, 3))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            self.get_logger().warn(f'VO: image decode error: {e}', throttle_duration_sec=5.0)
            return

        # Downsample to 640×480 for speed
        gray = cv2.resize(gray, (640, 480))
        self._process_frame(gray)

    # ── VO core ──────────────────────────────────────────────────────────────

    def _process_frame(self, gray: np.ndarray):
        with self._lock:
            height = self._height

        if self._prev_gray is None:
            # Bootstrap: only at flight altitude so feature scale matches tracking altitude
            if height < 0.9:
                self.get_logger().info(
                    f'VO: waiting for altitude (h={height:.2f}m < 0.9m)', throttle_duration_sec=2.0)
                return  # wait until airborne — ground-level features can't be tracked at 1.5m
            kp = self._orb.detect(gray, None)
            if len(kp) < MIN_FEATURES:
                self.get_logger().warn(
                    f'VO: not enough features at h={height:.2f}m ({len(kp)} < {MIN_FEATURES})',
                    throttle_duration_sec=2.0)
                return  # retry next frame at altitude
            pts = np.array([[k.pt] for k in kp], dtype=np.float32)
            self._prev_gray = gray
            self._prev_pts  = pts
            # Save first good frame for visual inspection
            if not self._debug_saved:
                cv2.imwrite('/tmp/vo_debug_frame.png', gray)
                self.get_logger().info(f'VO: saved debug frame /tmp/vo_debug_frame.png ({len(kp)} ORB features)')
                self._debug_saved = True
            return

        # Track features with Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if curr_pts is None:
            self._reset_features(gray)
            return

        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < MIN_FEATURES:
            self._track_fail += 1
            self.get_logger().warn(
                f'VO: low feature count {len(good_prev)} — resetting', throttle_duration_sec=3.0)
            self._reset_features(gray)
            return
        self._track_fail = 0

        # Skip accumulation on first frame after reset — LK flow between freshly
        # re-detected points and the next frame can be spuriously large, which
        # would inject a false position jump into EKF2 and destabilize attitude.
        if self._just_reset:
            self._just_reset = False
            self._prev_gray = gray
            self._prev_pts  = good_curr.reshape(-1, 1, 2)
            return

        # Only accumulate when airborne — floor noise at h<0.5m causes EKF2 to
        # flag "horizontal position unstable" and block arming.
        if height < 0.5:
            self._prev_gray = gray
            self._prev_pts  = good_curr.reshape(-1, 1, 2)
            with self._lock:
                self._vel_valid = False  # don't publish zero velocity to EKF2
            return

        # Direct optical flow → world displacement.
        # Essential Matrix t_unit is always unit-length (scale ambiguity), so
        # "t_unit * altitude" is physically wrong (3m/frame at h=3m).
        # Correct physics: world_d = -flow_pixels * altitude / focal_length_ds.
        FX_DS = FX * 0.5   # focal length for 640×480 (half of 1280×960)
        FY_DS = FY * 0.5

        flow = good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2)  # (N,2) pixels

        # Reject outliers: keep flows within 3 MADs of the median per axis
        med = np.median(flow, axis=0)
        mad = np.median(np.abs(flow - med), axis=0) + 1e-6
        inlier_mask = np.all(np.abs(flow - med) < 3.0 * mad, axis=1)
        if inlier_mask.sum() < MIN_FEATURES // 2:
            self._reset_features(gray)
            return

        mean_flow = np.median(flow[inlier_mask], axis=0)
        # Sign convention for downward-facing camera (pitch=+90° from body):
        # Empirically verified: image +u corresponds to ENU West, image +v to ENU South.
        # Drone moves East → features move West → flow_u +ve → dx_cam = +flow_u = +East ✓
        # Drone moves North → features move South → flow_v +ve → dy_cam = +flow_v = +North ✓
        dx_cam = +float(mean_flow[0]) * height / FX_DS
        dy_cam = +float(mean_flow[1]) * height / FY_DS

        # Log flow values every 30 frames to diagnose tracking quality
        if self._frame_count % 60 == 0:
            self.get_logger().info(
                f'VO: flow=({mean_flow[0]:.3f},{mean_flow[1]:.3f})px '
                f'dxy=({dx_cam*100:.1f},{dy_cam*100:.1f})cm '
                f'h={height:.2f}m inliers={inlier_mask.sum()}/{len(flow)}')

        # Accumulate position + store instantaneous velocity
        with self._lock:
            self._pos_x += dx_cam
            self._pos_y += dy_cam
            self._vel_x   = dx_cam * VO_HZ   # displacement/frame → m/s
            self._vel_y   = dy_cam * VO_HZ
            self._vel_valid = True

            # Sanity bound — reset if drone supposedly left arena
            if abs(self._pos_x) > MAX_DRIFT_XY or abs(self._pos_y) > MAX_DRIFT_XY:
                self.get_logger().warn(
                    f'VO: position out of bounds ({self._pos_x:.1f},{self._pos_y:.1f}) — resetting to base')
                self._pos_x = BASE_X
                self._pos_y = BASE_Y

        # Refresh keypoints
        self._prev_gray = gray
        self._prev_pts  = good_curr[inlier_mask].reshape(-1, 1, 2)

    def _reset_features(self, gray: np.ndarray):
        kp = self._orb.detect(gray, None)
        if kp:
            self._prev_pts = np.array([[k.pt] for k in kp], dtype=np.float32)
        self._prev_gray  = gray
        self._just_reset = True   # suppress accumulation on next frame
        self._pause_frames = 3    # hold last known pose, don't publish for 3 frames

    # ── ROS publish callback ─────────────────────────────────────────────────

    def _publish_cb(self):
        with self._lock:
            if self._pause_frames > 0:
                self._pause_frames -= 1
                return
            if not self._vel_valid:
                return  # don't poison EKF2 with zero velocity during ground-hold
            x, y, z = self._pos_x, self._pos_y, self._height
            vx, vy = self._vel_x, self._vel_y

        now = self.get_clock().now().to_msg()

        # Position (diagnostic only — EKF2_EV_CTRL=4 ignores position bit)
        ps = PoseStamped()
        ps.header.stamp    = now
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        self._pub.publish(ps)

        # Velocity (EKF2_EV_CTRL=4 — EKF2 fuses this, drift-free)
        ts = TwistStamped()
        ts.header.stamp    = now
        ts.header.frame_id = 'map'
        ts.twist.linear.x  = vx
        ts.twist.linear.y  = vy
        ts.twist.linear.z  = 0.0   # altitude from rangefinder, not VO
        self._vel_pub.publish(ts)

        self._pub_count += 1
        if self._pub_count % int(VO_HZ * 5) == 1:
            self.get_logger().info(
                f'VO pos=({x:.3f},{y:.3f},{z:.3f})  '
                f'vel=({vx:.2f},{vy:.2f})m/s  '
                f'h={z:.2f}m  frames={self._frame_count}  fails={self._track_fail}')


def main():
    rclpy.init()
    node = VOPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
