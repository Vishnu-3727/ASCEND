# Runtime Execution Guide

## Terminal 1 — MAVProxy

Activate virtual environment:

```bash
source ~/ijro/ascend/bin/activate
```

Start MAVProxy:

```bash
mavproxy.py \
--master=/dev/ttyAMA0,921600 \
--out=udp:127.0.0.1:14551 \
--out=udp:<LAPTOP_IP>:14550
```

Purpose:

* Receives MAVLink from Cube Orange through TELEM1 UART.
* Forwards MAVLink to MAVROS on localhost UDP 14551.
* Forwards telemetry to QGroundControl on the laptop through UDP 14550.

---

## Terminal 2 — MAVROS

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=77

ros2 launch mavros px4.launch \
fcu_url:=udp://127.0.0.1:14551@
```

Purpose:

* Receives MAVLink from MAVProxy.
* Publishes ROS topics and services.
* Acts as the ROS ↔ PX4 bridge.

---

## Terminal 3 — Verify MAVROS Connection

```bash
source /opt/ros/jazzy/setup.bash

ros2 topic echo /mavros/state
```

Expected:

```yaml
connected: true
armed: false
```

If `connected: true`, communication between PX4 and MAVROS is established.

---

## Terminal 4 — Mission Script

```bash
source /opt/ros/jazzy/setup.bash
source ~/ijro/ascend/bin/activate

cd ~/ijro/ASCEND
```

Run the required mission:

```bash
python takeoff.py
```

or

```bash
python survey.py
```

or

```bash
python landing.py
```

or

```bash
python mission.py
```

---

## QGroundControl

1. Connect laptop to the same router/network as the Raspberry Pi.
2. Open QGroundControl.
3. MAVProxy forwards telemetry to UDP port 14550.
4. QGroundControl should automatically connect and display:

* Vehicle Status
* Flight Mode
* GPS
* Battery
* Telemetry

No manual serial port configuration is required.

---

# Communication Architecture

```text
Cube Orange (PX4)
        │
        │ UART (TELEM1)
        ▼
     MAVProxy
        │
        ├── UDP 14551 ──► MAVROS
        │                    │
        │                    ▼
        │              Mission Scripts
        │
        └── UDP 14550 ──► QGroundControl
```

---

# Mission Startup Sequence

1. Power Cube Orange.
2. Power Raspberry Pi.
3. Connect laptop to router.
4. SSH into Raspberry Pi.
5. Start MAVProxy.
6. Start MAVROS.
7. Verify `/mavros/state` shows `connected: true`.
8. Open QGroundControl and verify vehicle connection.
9. Run mission script.
