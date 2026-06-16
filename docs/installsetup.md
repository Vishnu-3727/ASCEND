# INSTALL_SETUP.md

## 1. Update System

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

---

## 2. Install ROS 2 Jazzy

```bash
sudo apt install software-properties-common -y

sudo add-apt-repository universe -y

sudo apt update
```

Add ROS repository:

```bash
sudo curl -sSL \
https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
-o /usr/share/keyrings/ros-archive-keyring.gpg
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/ros2.list
```

```bash
sudo apt update
```

Install ROS:

```bash
sudo apt install -y ros-jazzy-base
```

Configure ROS:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

source ~/.bashrc
```

---

## 3. Install MAVROS

```bash
sudo apt install -y \
ros-jazzy-mavros \
ros-jazzy-mavros-extras \
ros-jazzy-mavros-msgs
```

Verify:

```bash
ros2 pkg list | grep mavros
```

Expected:

```text
mavros
mavros_extras
mavros_msgs
```

---

## 4. Install GeographicLib Datasets

Required by MAVROS.

```bash
sudo /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh
```

Verify:

```bash
ls /usr/share/GeographicLib/geoids/
```

Expected:

```text
egm96-5.pgm
```

---

## 5. Create Workspace Directory

```bash
mkdir -p ~/ijro

cd ~/ijro
```

---

## 6. Create Python Virtual Environment

```bash
python3 -m venv ascend
```

Activate:

```bash
source ~/ijro/ascend/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 7. Install Python Dependencies

Inside virtual environment:

```bash
pip install \
numpy \
opencv-python-headless \
pymavlink \
pyserial \
pyyaml \
MAVProxy
```

---

## 8. Clone Repository

```bash
cd ~/ijro

git clone <REPOSITORY_URL> ASCEND
```

Install project requirements:

```bash
source ~/ijro/ascend/bin/activate

cd ~/ijro/ASCEND

pip install -r requirements.txt
```

---

## 9. Configure ROS Domain ID

```bash
echo "export ROS_DOMAIN_ID=77" >> ~/.bashrc

source ~/.bashrc
```

Verify:

```bash
echo $ROS_DOMAIN_ID
```

Expected:

```text
77
```

---

## 10. UART Permissions

```bash
sudo usermod -aG dialout $USER
```

Logout and login again.

Temporary test:

```bash
sudo chmod 666 /dev/ttyAMA0
```

---

## 11. Verify Cube Connection

```bash
source ~/ijro/ascend/bin/activate

python - << 'EOF'
from pymavlink import mavutil

master = mavutil.mavlink_connection(
    '/dev/ttyAMA0',
    baud=921600
)

master.wait_heartbeat(timeout=15)

print("Heartbeat received!")
print("System:", master.target_system)
EOF
```

Expected:

```text
Heartbeat received!
```

