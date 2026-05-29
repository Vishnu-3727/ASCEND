# ASCEND IRoC-U 2026 — Drone Autonomy Stack

GPS-denied autonomous arena exploration with precision landing on a Raspberry Pi + Pixhawk.
This workspace is the **real-hardware target** — all SITL scripts are excluded from this guide.

---

## Repo Structure

```
drone_ws_phase3/
├── src/
│   ├── exploration_mission_poly.py   # Main mission (run this for full flight)
│   ├── landing_pad_detector.py       # Pad detection — needs 1 change for real HW (see below)
│   ├── hover_test.py                 # First-boot sanity check
│   ├── axis_cal_mission.py           # Optical flow axis calibration
│   ├── vo_publisher.py               # Visual odometry via downward camera (ORB tracking)
│   ├── vio_publisher.py              # VIO publisher (use if you have a VIO sensor)
│   └── offboard_mission.py           # Bare-bones offboard baseline for debugging
├── arenas/
│   ├── rect_baseline.json            # 10.67 × 7.62 m rectangle
│   ├── hex_regular.json
│   ├── pent_irregular.json
│   └── hex_irregular.json
└── tools/
    └── gen_arena.py                  # Generate a new arena JSON + SDF (SITL only)
```

---

## Prerequisites

### On the Raspberry Pi

```bash
# ROS 2 Humble
# MAVROS
sudo apt install ros-humble-mavros ros-humble-mavros-extras
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# Python deps
pip3 install opencv-python numpy
```

### Hardware connections

| Component | Connection |
|---|---|
| Pixhawk (TELEM2) | Pi UART or USB-serial (`/dev/ttyAMA0` or `/dev/ttyUSB0`) |
| Downward camera | Pi CSI or USB (`/dev/video0`) |
| Optical flow sensor (PMW3901 / PX4FLOW) | Pixhawk I2C or UART |
| Rangefinder (LW20 / TFmini) | Pixhawk UART |

---

## Pixhawk Parameters (set via QGroundControl once)

```
EKF2_EV_CTRL     = 0        # pure optical flow, no VIO
EKF2_AID_MASK    = 2        # optical flow only
SENS_FLOW_ROT    = ?        # set after axis calibration (step 1 below)
MPC_XY_VEL_MAX   = 0.6      # matches MAX_XY_SPEED in the script
MPC_Z_VEL_MAX_UP = 0.5
COM_RCL_EXCEPT   = 4        # offboard loss → hold, not RTL
```

---

## Step-by-Step: First Real Flight

### Step 1 — Axis calibration (do this once per new airframe)

Determines the correct `SENS_FLOW_ROT` value for your optical flow sensor mount.

```bash
ros2 run mavros mavros_node --ros-args -p fcu_url:=/dev/ttyAMA0:921600 &
python3 src/axis_cal_mission.py
```

The drone arms, climbs to 2 m, then flies: hold 10 s → North 5 s → hold → East 5 s → hold → South 5 s → land.
Watch EKF local position in `rostopic echo /mavros/local_position/pose`.
If North velocity produces +Y EKF drift → `SENS_FLOW_ROT = 0`. Adjust until axes align. Set in QGC.

---

### Step 2 — Hover test (do this before every mission day)

Validates that optical flow is stable before committing to exploration.

```bash
python3 src/hover_test.py
```

Sequence: PRESTREAM 6 s → TAKEOFF → HOLD 60 s @ 2 m → LAND → report.

**Pass:** EKF (x, y) stays within ±0.15 m throughout.
**Drift warning:** any sample outside ±0.15 m — logged with axis and direction.
**Abort:** EKF exits ±0.50 m → immediate AUTO.LAND.

If drift > 0.15 m: check `SENS_FLOW_ROT`, sensor cleanliness, and lighting.

---

### Step 3 — Before Swapping in the AprilTag Detector

The `landing_pad_detector.py` shipped here uses **HSV red-blob detection** (SITL default).
For real hardware you need to swap to AprilTag.

**Print:** AprilTag36h11 ID 0, at least 30 cm wide. Place flat at your base station.

**Edit `src/landing_pad_detector.py`** — find the blob detection block and replace with:

```python
# REAL HARDWARE: swap HSV blob for AprilTag
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

corners, ids, _ = detector.detectMarkers(gray)
if ids is not None and 0 in ids:
    idx    = list(ids.flatten()).index(0)
    c      = corners[idx][0]
    u, v   = c.mean(axis=0)          # centroid pixel
    # then compute dx_body / dy_body the same way as the blob path
```

Everything else (altitude scaling, `/landing_pad/offset` topic, PAD_SEARCH state) stays identical.

---

### Step 4 — Define your arena

Copy an existing arena JSON and edit the polygon coordinates to match your real arena.

```json
{
  "name": "my_arena",
  "polygon": [[x0,y0], [x1,y1], ...],   // 3–6 vertices, ENU metres, base station at (0,0)
  "poly_margin": 0.50,                   // inner keep-out from walls (m)
  "base": [0.0, 0.0],                    // landing pad position
  "spawn": [0.0, 0.0, 0.20],
  "spawn_yaw": 0.0,
  "marker_grid_res": 1.0                 // exploration grid cell size (m)
}
```

Place the drone at the base station corner. That point becomes EKF origin (0, 0).

---

### Step 5 — Run the full mission

```bash
# Terminal 1: MAVROS
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=/dev/ttyAMA0:921600 \
  -p gcs_url:=udp://@YOUR_GCS_IP:14550

# Terminal 2: Landing pad detector
python3 src/landing_pad_detector.py

# Terminal 3: Main mission
ARENA_CONFIG=arenas/my_arena.json python3 src/exploration_mission_poly.py
```

**State machine:**
```
IDLE → PRESTREAM (6 s, EKF warm-up) → TAKEOFF → EXPLORE (frontier grid)
     → RETURN → PAD_SEARCH (hover 2.2 m, acquire pad) → DESCEND (visual servo)
     → LAND → DONE
```

---

## Key Flight Parameters (in `exploration_mission_poly.py`)

| Parameter | Default | Notes |
|---|---|---|
| `FLIGHT_ALT` | 1.5 m | Lower = better optical flow SNR |
| `MAX_XY_SPEED` | 0.6 m/s | Reduce to 0.3 for first outdoor test |
| `WP_RADIUS` | 0.45 m | Waypoint acceptance radius |
| `PRESTREAM_S` | 6.0 s | EKF warm-up before arming |
| `UNSTABLE_ANGLE_DEG` | 30° | Attitude fault threshold |
| `PRECLAND_ENABLED` | True | Set False to fall back to AUTO.LAND |

---

## Safety Layers

1. **Waypoint gating** — all exploration waypoints are validated inside the inner polygon before being set as goals.
2. **Brake zone** — carrot speed reduces near polygon boundary.
3. **Hard freeze** — carrot frozen if it would exit the inner polygon.
4. **Attitude fault** → `STABILIZE` state → resume saved phase on recovery.
5. **Hover test abort** — auto-land if EKF drift > 0.50 m.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Drone drifts after takeoff | Wrong `SENS_FLOW_ROT` — redo axis cal |
| PAD_SEARCH never acquires pad | AprilTag too small or lighting; check `/landing_pad/offset` topic |
| EKF heading jumps | `SENS_FLOW_ROT` or magnetometer interference from motors |
| Drone hugs one wall | `poly_margin` too small for actual arena dimensions |
| OFFBOARD rejected | MAVROS not connected — check `fcu_url` and baud rate |

---

## What NOT to Run on Pi

The following are **SITL-only** — they launch Gazebo and serve no purpose on hardware:

- `run_sitl.sh`, `launch_sitl.sh`, `run_px4.sh`, `stop_sitl.sh`
- `worlds/`, `models/` directories
- `tools/gen_arena.py` (SDF generation — arena JSON is still used)
