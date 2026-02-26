"""
gym_gazebo_env.py

A synchronous Gymnasium environment that controls an Ignition/Gazebo (ros_gz) simulation.

Features:
- Publishes velocity commands (geometry_msgs/Twist) to a cmd topic.
- Subscribes to odometry (nav_msgs/Odometry) for observations.
- Advances the simulator deterministically using the ros_gz `ControlWorld` service:
    /world/<world_name>/control (ros_gz_interfaces/srv/ControlWorld)
- On world reset, respawns the robot using the ros_gz_sim `create` CLI (fallback),
  matching the spawn used in the launch file.

Prerequisites:
- ROS 2 (Jazzy or compatible) and your workspace sourced in the shell that runs this Python process
  so that `ros2` CLI is available on PATH (required for spawn fallback).
- ros_gz_bridge running with a service bridge for /world/<world>/control:
    ros2 run ros_gz_bridge parameter_bridge /world/<world>/control@ros_gz_interfaces/srv/ControlWorld
- The script expects odometry on the topic you pass (default: /wheel_encoder/odom).
- gymnasium installed in the Python environment that runs this script.

Design notes:
- The environment uses synchronous stepping (`ControlWorld` with `multi_step`) for deterministic control.
- Reset performs a world reset (reset_all) and then respawns the robot using the same CLI used at launch.
  This keeps the SDF/world file unchanged and allows runtime-spawned models to be re-created.
"""

from __future__ import annotations

import time
import threading
import subprocess
import shutil
import os
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldControl, WorldReset

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------
_SERVICE_TIMEOUT = 10.0        # seconds waiting for the ControlWorld service call
_OBS_WAIT_TIMEOUT = 2.0        # seconds to wait for a fresh odometry update
_SPAWN_WAIT_TIMEOUT = 6.0      # seconds to wait for odometry after spawning the robot


# -----------------------------------------------------------------------------
# Helper class: single rclpy node with background spinning
# -----------------------------------------------------------------------------
class _RosNodeHolder:
    """
    Lightweight helper that creates a single rclpy Node and spins it in a background
    thread so the environment can use publishers, subscribers and service clients
    from a synchronous Python thread.

    The holder initializes rclpy if necessary; calling `shutdown()` will destroy the
    node and attempt to shutdown rclpy (cautious if other code shares rclpy).
    """

    def __init__(self, node_name: str = 'gym_gazebo_node'):
        # Initialize rclpy once per process if not already initialized.
        if not rclpy.ok():
            rclpy.init(args=None)

        # Create the ROS node used for publishers/subscriptions/service clients.
        self.node: Node = Node(node_name)

        # Start a background thread that runs rclpy.spin(node).
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

    def _spin(self) -> None:
        """Background spinner — keeps the node responsive to callbacks."""
        try:
            rclpy.spin(self.node)
        except Exception:
            # Defensive: ignore shutdown/interrupt exceptions in the spin loop.
            pass

    def shutdown(self) -> None:
        """
        Clean up the node and try to shutdown rclpy. If your process runs multiple
        nodes or other components depending on rclpy, be careful — rclpy.shutdown()
        is called here defensively and may fail if other pieces still expect rclpy.
        """
        try:
            self.node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            # swallowing potential exceptions ensures close() can be called safely
            pass


# -----------------------------------------------------------------------------
# Gym environment: GazeboEnv
# -----------------------------------------------------------------------------
class GazeboEnv(gym.Env):
    """
    A Gymnasium environment wrapping a ros_gz (Ignition Gazebo) simulation.

    Observation space:
        numpy array shape (5,) -> [x, y, yaw, vx, vyaw] (float32)

    Action space:
        2-D continuous -> [linear_velocity (m/s), angular_velocity (rad/s)]

    Parameters
    ----------
    world_name : str
        Name of the Gazebo world (used to build the ControlWorld service name).
        Default: 'depot'.
    cmd_topic : str
        Topic to publish geometry_msgs/Twist commands to (default '/cmd_vel').
    odom_topic : str
        Topic to subscribe for odometry (default '/wheel_encoder/odom').
    sim_steps_per_env_step : int
        How many simulator physics steps to advance per Env.step() call; passed as
        `multi_step` in ControlWorld. Raising this reduces service-call overhead at
        the cost of lower agent action frequency.
    spawn_name : str
        Name used when spawning the robot (CLI create) after reset.
    spawn_pose : tuple[float, float, float]
        (x, y, z) pose used when respawning the robot.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        world_name: str = 'depot',
        cmd_topic: str = '/cmd_vel',
        odom_topic: str = '/wheel_encoder/odom',
        sim_steps_per_env_step: int = 1,
        spawn_name: str = 'robot',
        spawn_pose: Tuple[float, float, float] = (0.0, 0.0, 0.9)
    ):
        super().__init__()

        # Create or reuse a background rclpy node holder
        self._ros = _RosNodeHolder(node_name='gym_gazebo_env_node')

        # Simulation control parameters
        self._world = world_name
        self._control_service_name = f'/world/{self._world}/control'
        self._sim_steps_per_env_step = int(sim_steps_per_env_step)

        # Spawn parameters (used after reset to respawn models removed by world reset)
        self._spawn_name = spawn_name
        self._spawn_pose = spawn_pose

        # Publisher: velocity commands
        self._cmd_pub = self._ros.node.create_publisher(Twist, cmd_topic, 10)

        # QoS for odometry subscription: reliable and keep-last
        qos_odom = QoSProfile(depth=10)
        qos_odom.reliability = QoSReliabilityPolicy.RELIABLE
        qos_odom.history = QoSHistoryPolicy.KEEP_LAST

        # Internal odometry state and synchronization
        self._last_odom: Optional[np.ndarray] = None
        self._odom_lock = threading.Lock()
        self._odom_timestamp: float = 0.0

        # Subscribe to odometry from the provided topic
        self._odom_sub = self._ros.node.create_subscription(
            Odometry, odom_topic, self._odom_cb, qos_odom
        )

        # Service client for stepping / pausing / resetting the sim
        self._control_client = self._ros.node.create_client(ControlWorld, self._control_service_name)
        if not self._control_client.wait_for_service(timeout_sec=_SERVICE_TIMEOUT):
            raise RuntimeError(
                f"Timeout waiting for service {self._control_service_name}. "
                "Ensure ros_gz_bridge is running and bridging the ControlWorld service."
            )

        # Gym spaces
        self.action_space = spaces.Box(low=np.array([-1.0, -3.14]), high=np.array([1.0, 3.14]), dtype=np.float32)
        obs_high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Last-observation fallback
        self._last_obs = np.zeros(5, dtype=np.float32)

    # ----------------------------
    # Internal callback / helpers
    # ----------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        """
        Odometry callback. Converts nav_msgs/Odometry into the internal observation
        vector [x, y, yaw, vx, vyaw] and stores a timestamp so we can detect fresh updates.
        """
        with self._odom_lock:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            # minimal quaternion -> yaw conversion (sufficient for planar robots)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = float(np.arctan2(siny_cosp, cosy_cosp))

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.angular.z

            self._last_odom = np.array([px, py, yaw, vx, vy], dtype=np.float32)
            self._odom_timestamp = time.time()

    def _publish_action(self, action: np.ndarray) -> None:
        """
        Publish action (linear, angular) as geometry_msgs/Twist on the cmd topic.
        """
        t = Twist()
        t.linear.x = float(action[0])
        t.angular.z = float(action[1])
        self._cmd_pub.publish(t)

    def _call_world_control(
        self,
        *,
        pause: bool = False,
        step: bool = False,
        multi_step: int = 0,
        reset_all: bool = False,
        timeout: float = 5.0
    ):
        """
        Call the ros_gz ControlWorld service with the requested options.

        Parameters
        ----------
        pause, step : bool
            Request to pause or step the simulator.
        multi_step : int
            Number of internal physics steps to run (if >0).
        reset_all : bool
            If True, request a full world reset (this clears runtime-spawned models).
        timeout : float
            Seconds to wait for the service response.

        Returns
        -------
        Service response object
        """
        req = ControlWorld.Request()
        wc = WorldControl()
        wc.pause = bool(pause)
        wc.step = bool(step)
        if multi_step:
            wc.multi_step = int(multi_step)
        if reset_all:
            wr = WorldReset()
            wr.all = True
            wc.reset = wr
        req.world_control = wc

        future = self._control_client.call_async(req)
        t0 = time.time()
        # Busy-wait loop with a short sleep — acceptable here because the environment is
        # already synchronous and we need deterministic behavior.
        while rclpy.ok() and not future.done() and (time.time() - t0) < timeout:
            time.sleep(0.001)

        if not future.done():
            raise RuntimeError("ControlWorld service call timed out.")
        return future.result()

    def _wait_for_obs_update(self, timeout: float = _OBS_WAIT_TIMEOUT) -> Optional[np.ndarray]:
        """
        Wait for a new odometry update. Returns the most recent odometry vector if
        one arrives within `timeout` seconds, otherwise returns the last known odom
        or None if never seen.

        This is used to synchronously wait for sensors after stepping the sim.
        """
        t0 = time.time()
        initial_ts = self._odom_timestamp
        while (time.time() - t0) < timeout:
            with self._odom_lock:
                if self._odom_timestamp != initial_ts and self._last_odom is not None:
                    return self._last_odom.copy()
            time.sleep(0.001)

        with self._odom_lock:
            return None if self._last_odom is None else self._last_odom.copy()

    def _spawn_robot_cli(self, name: Optional[str] = None, pose: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Respawn the robot using the `ros2 run ros_gz_sim create ...` CLI.

        This fallback is used after calling world reset (reset_all=True) because
        a world reset removes runtime-spawned models. Using the same CLI as the
        initial launch keeps behavior identical.

        Returns True if the CLI returns exit code 0, False otherwise.

        Requirements:
        - `ros2` CLI must be available on PATH.
        - The process running this code must have a sourced ROS environment so that
          the `ros2` command works as expected.
        """
        name = name or self._spawn_name
        pose = pose or self._spawn_pose

        ros2_bin = shutil.which("ros2")
        if ros2_bin is None:
            print("[GazeboEnv] spawn failed: 'ros2' not found on PATH")
            return False

        x, y, z = pose
        cmd = [
            ros2_bin, "run", "ros_gz_sim", "create",
            "-name", name,
            "-topic", "/robot_description",
            "-x", str(x),
            "-y", str(y),
            "-z", str(z)
        ]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ, timeout=10.0)
            if proc.returncode == 0:
                print(f"[GazeboEnv] spawn CLI succeeded (name={name}).")
                return True
            else:
                stderr = proc.stderr.decode('utf-8', errors='ignore')
                print(f"[GazeboEnv] spawn CLI failed, return code {proc.returncode}. stderr:\n{stderr}")
                return False
        except Exception as e:
            print(f"[GazeboEnv] spawn CLI exception: {e}")
            return False

    # ----------------------------
    # Gym API
    # ----------------------------
    def step(self, action):
        """
        Apply an action and advance the simulation synchronously.

        Steps:
        1. Publish the velocity command.
        2. Call ControlWorld with multi_step=self._sim_steps_per_env_step and pause=True
           so the simulator advances deterministically and then stops again.
        3. Wait for odometry update and return obs, reward, done, truncated, info.

        Returns
        -------
        obs : np.ndarray
            Observation vector [x, y, yaw, vx, vyaw]
        reward : float
            Example reward (negative Euclidean distance to origin). Replace with your reward.
        terminated : bool
        truncated : bool
        info : dict
        """
        action = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action), f"Action out of bounds: {action}"

        # 1) publish
        self._publish_action(action)

        # 2) step the sim synchronously (multi_step advances internal physics iterations)
        self._call_world_control(
            pause=True,
            multi_step=self._sim_steps_per_env_step,
            timeout=_SERVICE_TIMEOUT
        )

        # 3) wait for sensors (odom)
        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            # no fresh odom arrived; fall back to last observation
            obs = self._last_obs.copy()
        else:
            self._last_obs = obs

        # ----------------------------
        # Question, Design the Reward Function to follow the path
        # ----------------------------
        # Example reward: negative distance to origin (toy example; You must eplace it as needed)
        reward = -np.linalg.norm(obs[:2])
        terminated = False
        truncated = False
        # ----------------------------
        # Question end
        # ----------------------------
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment/episode.

        Strategy:
        - Perform a world reset (reset_all=True) — this restores SDF-included state
          but removes runtime-spawned models (such as the `create` spawned robot).
        - Respawn the robot using the CLI fallback (_spawn_robot_cli) so the runtime-spawned
          robot is present again.
        - Wait for odometry to appear and return the first observation.

        Returns
        -------
        obs : np.ndarray
            Initial observation for the new episode (and an empty info dict).
        """
        # 1) reset the world (this clears runtime-spawned entities)
        try:
            self._call_world_control(reset_all=True, timeout=_SERVICE_TIMEOUT)
        except Exception as e:
            print(f"[GazeboEnv] world reset call failed: {e}")

        # small delay to allow the world to settle
        time.sleep(0.2)

        # clear cached odom
        with self._odom_lock:
            self._last_odom = None
            self._odom_timestamp = 0.0

        # 2) respawn the robot using CLI fallback
        spawned = self._spawn_robot_cli(name=self._spawn_name, pose=self._spawn_pose)
        if spawned:
            # wait longer for odom after spawn
            obs = self._wait_for_obs_update(timeout=_SPAWN_WAIT_TIMEOUT)
            if obs is not None:
                self._last_obs = obs
                return obs, {}
            else:
                print("[GazeboEnv] spawn completed but no odom received within timeout.")
        else:
            print("[GazeboEnv] spawn attempt failed. Robot may not be present.")

        # 3) final fallback: maybe the robot was restored by the SDF reset; try a short wait
        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            obs = np.zeros(5, dtype=np.float32)
        self._last_obs = obs
        return obs, {}

    def close(self) -> None:
        """
        Cleanly shutdown the ROS node and resources used by the environment.
        """
        try:
            self._ros.shutdown()
        except Exception:
            pass
