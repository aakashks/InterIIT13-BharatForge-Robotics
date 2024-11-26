# Start from Ubuntu 22.04
FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Set up timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    xrdp \
    xfce4 \
    xfce4-terminal \
    firefox \
    dbus-x11 \
    xauth \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure XRDP
RUN adduser xrdp ssl-cert
RUN sed -i 's/3389/3390/g' /etc/xrdp/xrdp.ini
RUN echo "startxfce4" > /etc/skel/.xsession
RUN echo "startxfce4" > /root/.xsession

# Set up ROS2 Humble repositories
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Full
RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
    ros-humble-gazebo-* \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Set up ROS2 environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Create a new user
RUN useradd -m -s /bin/bash ros_user && \
    echo "ros_user:password" | chpasswd && \
    adduser ros_user sudo

# Set up ROS2 environment for ros_user
RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros_user/.bashrc

RUN pip install uv

# Create startup script
RUN echo '#!/bin/bash\n\
service dbus start\n\
service xrdp start\n\
tail -f /dev/null' > /start.sh && \
chmod +x /start.sh

# Expose XRDP port
EXPOSE 3390

# Set the default command
CMD ["/start.sh"]