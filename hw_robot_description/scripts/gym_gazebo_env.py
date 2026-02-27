"""
gym_gazebo_env.py (AMCL Version)

Changes for AMCL:
- Subscribes to geometry_msgs/PoseWithCovarianceStamped instead of nav_msgs/Odometry.
- Calculates velocity (vx, vyaw) manually via finite difference since AMCL doesn't provide twist.
- Default topic changed to '/amcl_pose'.
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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped # <--- تغییر ۱: ایمپورت پیام جدید
# from nav_msgs.msg import Odometry # حذف شد

from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldControl, WorldReset

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------
_SERVICE_TIMEOUT = 10.0
_OBS_WAIT_TIMEOUT = 2.0
_SPAWN_WAIT_TIMEOUT = 6.0


# -----------------------------------------------------------------------------
# Helper class: single rclpy node with background spinning
# -----------------------------------------------------------------------------
class _RosNodeHolder:
    def __init__(self, node_name: str = 'gym_gazebo_node'):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node: Node = Node(node_name)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

    def _spin(self) -> None:
        try:
            rclpy.spin(self.node)
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Gym environment: GazeboEnv
# -----------------------------------------------------------------------------
class GazeboEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        world_name: str = 'depot',
        cmd_topic: str = '/cmd_vel',
        odom_topic: str = '/amcl_pose', # <--- تغییر ۲: تاپیک پیش‌فرض AMCL
        sim_steps_per_env_step: int = 1,
        spawn_name: str = 'robot',
        spawn_pose: Tuple[float, float, float] = (0.0, 0.0, 0.9)
    ):
        super().__init__()

        self._ros = _RosNodeHolder(node_name='gym_gazebo_env_node')

        self._world = world_name
        self._control_service_name = f'/world/{self._world}/control'
        self._sim_steps_per_env_step = int(sim_steps_per_env_step)

        self._spawn_name = spawn_name
        self._spawn_pose = spawn_pose

        self._cmd_pub = self._ros.node.create_publisher(Twist, cmd_topic, 10)

        # <--- تغییر ۳: تنظیم QoS مناسب برای AMCL
        qos_amcl = QoSProfile(depth=10)
        qos_amcl.reliability = QoSReliabilityPolicy.RELIABLE
        qos_amcl.durability = DurabilityPolicy.TRANSIENT_LOCAL # معمولاً AMCL اینطور است
        qos_amcl.history = QoSHistoryPolicy.KEEP_LAST

        self._last_odom: Optional[np.ndarray] = None
        self._odom_lock = threading.Lock()
        self._odom_timestamp: float = 0.0
        
        # متغیرهای کمکی برای محاسبه سرعت (چون AMCL سرعت نمی‌دهد)
        self._prev_pose_time = None
        self._prev_pose_coords = None

        # <--- تغییر ۴: تغییر نوع سابسکرایب به PoseWithCovarianceStamped
        self._odom_sub = self._ros.node.create_subscription(
            PoseWithCovarianceStamped, odom_topic, self._odom_cb, qos_amcl
        )

        self._control_client = self._ros.node.create_client(ControlWorld, self._control_service_name)
        if not self._control_client.wait_for_service(timeout_sec=_SERVICE_TIMEOUT):
            raise RuntimeError(
                f"Timeout waiting for service {self._control_service_name}."
            )

        self.action_space = spaces.Box(low=np.array([-1.0, -3.14]), high=np.array([1.0, 3.14]), dtype=np.float32)
        obs_high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._last_obs = np.zeros(5, dtype=np.float32)

    # ----------------------------
    # Internal callback / helpers
    # ----------------------------
    def _odom_cb(self, msg: PoseWithCovarianceStamped) -> None: # <--- تغییر ۵: تغییر ورودی تابع
        """
        Callback for AMCL pose. Calculates velocity manually.
        """
        with self._odom_lock:
            current_time = time.time()
            
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = float(np.arctan2(siny_cosp, cosy_cosp))

            # <--- تغییر ۶: محاسبه سرعت دستی (Finite Difference)
            vx = 0.0
            vyaw = 0.0
            
            if self._prev_pose_time is not None:
                dt = current_time - self._prev_pose_time
                if dt > 0.0001: # جلوگیری از تقسیم بر صفر
                    # محاسبه جابجایی
                    dx = px - self._prev_pose_coords[0]
                    dy = py - self._prev_pose_coords[1]
                    dyaw = yaw - self._prev_pose_coords[2]

                    # نرمال‌سازی تغییر زاویه (بین -pi و pi)
                    while dyaw > np.pi: dyaw -= 2*np.pi
                    while dyaw < -np.pi: dyaw += 2*np.pi
                    
                    # محاسبه سرعت تقریبی
                    dist = np.sqrt(dx**2 + dy**2)
                    # جهت حرکت (مثبت یا منفی بودن سرعت خطی)
                    # اگر زاویه حرکت با yaw فعلی هم‌جهت بود مثبت، وگرنه منفی
                    motion_yaw = np.arctan2(dy, dx)
                    yaw_diff = abs(motion_yaw - yaw)
                    while yaw_diff > np.pi: yaw_diff -= 2*np.pi
                    yaw_diff = abs(yaw_diff)
                    
                    sign = 1.0
                    if yaw_diff > (np.pi / 2):
                        sign = -1.0
                    
                    vx = (dist / dt) * sign
                    vyaw = dyaw / dt

            # ذخیره مقادیر فعلی برای دور بعد
            self._prev_pose_coords = (px, py, yaw)
            self._prev_pose_time = current_time

            self._last_odom = np.array([px, py, yaw, vx, vyaw], dtype=np.float32)
            self._odom_timestamp = current_time

    def _publish_action(self, action: np.ndarray) -> None:
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
        while rclpy.ok() and not future.done() and (time.time() - t0) < timeout:
            time.sleep(0.001)

        if not future.done():
            raise RuntimeError("ControlWorld service call timed out.")
        return future.result()

    def _wait_for_obs_update(self, timeout: float = _OBS_WAIT_TIMEOUT) -> Optional[np.ndarray]:
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

    def step(self, action):
        # 1. اعمال اکشن و استپ شبیه‌سازی (مانند قبل)
        action = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action), f"Action out of bounds: {action}"

        self._publish_action(action)

        self._call_world_control(
            pause=True,
            multi_step=self._sim_steps_per_env_step,
            timeout=_SERVICE_TIMEOUT
        )

        # 2. دریافت Observation جدید
        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            obs = self._last_obs.copy()
        else:
            self._last_obs = obs
        
        # --- بخش جدید: محاسبه پاداش و وضعیت ---
        
        # استخراج موقعیت فعلی ربات از obs (فرض: x=obs[0], y=obs[1])
        robot_pos = np.array(obs[:2])
        target_pos = np.array(self.target_waypoint) # باید در reset مقداردهی شده باشد

        # محاسبه فاصله تا هدف
        dist_to_target = np.linalg.norm(target_pos - robot_pos)

        # بررسی برخورد (نیاز به متد کمکی یا بررسی داده‌های لیزر دارد)
        # فرض: متدی دارید که اگر لیزر خیلی نزدیک بود True برمی‌گرداند
        is_collision = self._check_collision() 

        # تعریف مولفه‌های پاداش
        reward = 0.0
        r_arrival = 0.0
        r_collision = 0.0
        r_progress = 0.0
        r_time = -0.05  # جریمه زمانی ثابت برای تشویق به سرعت

        terminated = False
        truncated = False

        if is_collision:
            # سناریوی ۱: برخورد
            r_collision = -100.0
            reward = r_collision + r_time
            terminated = True
            
        elif dist_to_target < 0.2: 
            # سناریوی ۲: رسیدن به هدف (با تلورانس ۲۰ سانتی‌متر)
            r_arrival = 100.0
            reward = r_arrival + r_time
            terminated = True
            
        else:
            # سناریوی ۳: در حال حرکت به سمت هدف
            # محاسبه پیشرفت نسبت به مرحله قبل
            # self.prev_dist باید در reset مقداردهی اولیه شود
            diff = self.prev_dist - dist_to_target
            r_progress = 20.0 * diff  # ضریب ۲۰ برای تقویت اثر پیشرفت
            
            reward = r_progress + r_time

        # به‌روزرسانی فاصله برای گام بعدی
        self.prev_dist = dist_to_target

        # اطلاعات اضافی برای دیباگ
        info = {
            "dist_to_target": dist_to_target,
            "is_collision": is_collision,
            "reward_parts": {
                "progress": r_progress,
                "time": r_time,
                "arrival": r_arrival,
                "collision": r_collision
            }
        }

        return obs, float(reward), bool(terminated), bool(truncated), info


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            self._call_world_control(reset_all=True, timeout=_SERVICE_TIMEOUT)
        except Exception as e:
            print(f"[GazeboEnv] world reset call failed: {e}")

        time.sleep(0.2)
        
        # ریست کردن متغیرهای محاسبه سرعت
        self._prev_pose_time = None
        self._prev_pose_coords = None

        with self._odom_lock:
            self._last_odom = None
            self._odom_timestamp = 0.0

        spawned = self._spawn_robot_cli(name=self._spawn_name, pose=self._spawn_pose)
        if spawned:
            obs = self._wait_for_obs_update(timeout=_SPAWN_WAIT_TIMEOUT)
            if obs is not None:
                self._last_obs = obs
                return obs, {}
            else:
                print("[GazeboEnv] spawn completed but no pose received within timeout.")
        else:
            print("[GazeboEnv] spawn attempt failed. Robot may not be present.")

        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            obs = np.zeros(5, dtype=np.float32)
        self._last_obs = obs
        dist = np.linalg.norm([self.target_coord[0] - x, self.target_coord[1] - y])
        self.prev_dist_to_target = dist  # <--- این خط جدید اضافه شود
        return obs, {}

    def close(self) -> None:
        try:
            self._ros.shutdown()
        except Exception:
            pass
