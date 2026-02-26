# test_env.py
import numpy as np
from gym_gazebo_env import GazeboEnv
import time

env = GazeboEnv(world_name='depot', sim_steps_per_env_step=10)
obs, info = env.reset()

print("Initial obs:", obs)
for j in range(10):
    obs, info = env.reset()
    for i in range(500):
        action = np.array([0.2, 0.0], dtype=np.float32)  # move forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i:03d}: obs={obs}, reward={reward:.3f}")
        if terminated or truncated:
            break
        # no time.sleep() needed: stepping is synchronous with multi_step above

env.close()
