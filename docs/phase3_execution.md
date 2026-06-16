# Phase 3 – Autonomous Mission Execution

This document describes the complete startup sequence for executing autonomous missions using PX4, MAVProxy, MAVROS, ROS 2 Jazzy, and QGroundControl.

---

## Architecture

```text
Terminal 1
──────────
MAVProxy

Terminal 2
──────────
MAVROS

Terminal 3
──────────
Mission Script

Laptop
──────────
QGroundControl
```

```text
                    QGroundControl
                     10.77.0.10
                           ▲
                           │ UDP:14550
                           │
                    MAVProxy (Pi)
                           ▲
                           │ UDP:14551
                           │
                           ▼
                       MAVROS
                           ▲
                           │ ROS 2 Topics
                           │
                           ▼
                    Mission Script
                           ▲
                           │
                           ▼
                    PX4 Cube Orange
                      /dev/ttyAMA0
```

---

# Terminal 1 – MAVProxy

SSH into Raspberry Pi:

```bash
ssh px@10.77.0.20
```

Activate virtual environment:

```bash
source ~/ijro/ascend/bin/activate
```

Start MAVProxy:

```bash
mavproxy.py \
--master=/dev/ttyAMA0,921600 \
--out=udp:127.0.0.1:14551 \
--out=udp:10.77.0.10:14550
```

Expected output:

```text
online system 1
Received parameters
```

---

# Laptop – QGroundControl

Open QGroundControl.

QGC automatically listens on:

```text
UDP Port 14550
```

Verify:

```text
Vehicle Connected
GPS Lock
Battery Visible
Flight Mode Visible
```

---

# Terminal 2 – MAVROS

Open a new SSH session:

```bash
ssh px@10.77.0.20
```

Source ROS:

```bash
source /opt/ros/jazzy/setup.bash
```

Set ROS domain:

```bash
export ROS_DOMAIN_ID=77
```

Launch MAVROS:

```bash
ros2 launch mavros px4.launch \
fcu_url:=udp://127.0.0.1:14551@
```

Available topics:

```text
/mavros/state
/mavros/local_position/pose
/mavros/global_position/global
/mavros/imu/data
```

---

# Terminal 3 – Verify Connection

```bash
source /opt/ros/jazzy/setup.bash

ros2 topic echo /mavros/state
```

Expected:

```yaml
connected: true
armed: false
```

Proceed only if:

```yaml
connected: true
```

---

# Terminal 4 – Mission Execution

SSH into Raspberry Pi:

```bash
ssh px@10.77.0.20
```

Source ROS:

```bash
source /opt/ros/jazzy/setup.bash
```

Activate virtual environment:

```bash
source ~/ijro/ascend/bin/activate
```

Navigate to repository:

```bash
cd ~/ijro/ASCEND
```

Execute mission:

Takeoff:

```bash
python takeoff.py
```

Survey:

```bash
python survey.py
```

Full Mission:

```bash
python mission.py
```

---

# Mission Command Flow

```text
Mission Script
      │
      ▼
 MAVROS
      │
 UDP 14551
      │
      ▼
 MAVProxy
      │
 UART
      │
      ▼
 PX4 Cube Orange
```

Telemetry Return Path:

```text
PX4
 │
 ▼
Cube Orange
 │
 ▼
MAVProxy
 ├────────► QGroundControl
 │
 └────────► MAVROS
               │
               ▼
        Mission Script
```

---

# Pre-Flight Checklist

## MAVProxy

```bash
source ~/ijro/ascend/bin/activate

mavproxy.py \
--master=/dev/ttyAMA0,921600 \
--out=udp:127.0.0.1:14551 \
--out=udp:10.77.0.10:14550
```

## MAVROS

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=77

ros2 launch mavros px4.launch \
fcu_url:=udp://127.0.0.1:14551@
```

## Verify Connection

```bash
ros2 topic echo /mavros/state
```

Expected:

```yaml
connected: true
```

## Execute Mission

```bash
cd ~/ijro/ASCEND

python takeoff.py
```

## QGroundControl Checks

* Vehicle Connected
* GPS Lock
* Battery Healthy
* Position Estimate Available
* Correct Flight Mode

Only arm and execute missions after all checks pass.
