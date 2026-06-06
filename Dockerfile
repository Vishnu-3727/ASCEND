FROM ros:jazzy-ros-base

# Install MAVROS, OpenCV, and Python deps — no Gazebo
RUN apt-get update && apt-get install -y \
    ros-jazzy-mavros \
    ros-jazzy-mavros-extras \
    ros-jazzy-mavros-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    python3-opencv \
    python3-numpy \
    python3-pip \
    geographiclib-tools \
    && rm -rf /var/lib/apt/lists/*

# MAVROS needs GeographicLib datasets (EKF2 mag/world model)
RUN /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh

# Copy mission scripts and arena configs
WORKDIR /ascend
COPY drone_ws_phase3/src/ ./src/
COPY drone_ws_phase3/arenas/ ./arenas/

# Make scripts executable
RUN chmod +x src/*.py

# Source ROS2 on every shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]
