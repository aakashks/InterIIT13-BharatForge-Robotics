# Start from Ubuntu 22.04
FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Set up timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    xrdp \
    ubuntu-desktop \
    gnome-terminal \
    firefox \
    dbus-x11 \
    x11-apps \
    xterm \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Configure XRDP
RUN adduser xrdp ssl-cert
RUN echo "gnome-session" > /root/.xsession

RUN sed -i 's/3389/3390/g' /etc/xrdp/xrdp.ini && \
    sed -i 's/max_bpp=32/max_bpp=128/g' /etc/xrdp/xrdp.ini && \
    sed -i 's/xserverbpp=24/xserverbpp=128/g' /etc/xrdp/xrdp.ini && \
    echo "gnome-session" > /etc/skel/.xsession


# Set up ROS2 Humble repositories
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Full
RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
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
    echo "ros_user:ros_pass" | chpasswd && \
    usermod -aG sudo ros_user && \
    cp /etc/skel/.xsession /home/ros_user/.xsession && \
    chown ros_user:ros_user /home/ros_user/.xsession

# Set up ROS2 environment for ros_user
RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros_user/.bashrc
RUN echo "gnome-session" > /home/ros_user/.xsession

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