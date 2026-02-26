# robotic_course

Robotic course TA session packages. 

## Prerequisites
- ROS 2 installed and sourced (replace <ros_distro> with your distro, e.g. humble):
    ```
    source /opt/ros/<ros_distro>/setup.bash
    ```
- rosdep configured (run once):
    ```
    sudo apt update
    sudo apt install -y python3-rosdep
    sudo rosdep init        
    rosdep update
    ```

## Clone into your workspace
From your workspace root:
```
cd ~/robotics_ws/src
git clone git@github.com:FastaRobotics/robotic_course.git
```

## Install dependencies with rosdep
From the workspace root:
```
cd ~/robotics_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## Build
```
cd ~/robotics_ws
colcon build --symlink-install
```

## Source and run
After a successful build:
```
source ~/robotics_ws/install/setup.bash
# run nodes or launch files, for example:
# ros2 run <package> <executable>
# ros2 launch <package> <launch_file.launch.py>
```

## Troubleshooting
- If rosdep fails, inspect error messages and ensure required apt sources are available.
- If a dependency is a private repo, clone it into `src` before running rosdep or install it manually.

License: Apache License