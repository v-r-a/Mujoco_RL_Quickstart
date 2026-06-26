import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class FurutaEnv(gym.Env):
    """
    Standard Cartpole balancing environment using MuJoCo.
    Keeps the same class name so the user's training command history works out-of-the-box.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load the cartpole XML
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cartpole.xml"))
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space: sliding force on the cart [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observations: [cart_pos, cart_vel, pole_angle, pole_vel]
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -10.0, -0.418, -10.0]),
            high=np.array([4.8, 10.0, 0.418, 10.0]),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self._step_count = 0
        self._max_steps = 500  # 10 seconds at dt=0.02s

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],  # cart position
            self.data.qvel[0],  # cart velocity
            self.data.qpos[1],  # pole angle
            self.data.qvel[1]   # pole angular velocity
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial state slightly around upright position (qpos[1]=0 is upright)
        self.data.qpos[0] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[1] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[0] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[1] = self.np_random.uniform(-0.05, 0.05)
        
        mujoco.mj_forward(self.model, self.data)
        
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[0] = clipped_action[0]
        
        # Step simulation (2 substeps of 0.01s = 0.02s per control step)
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
            
        self._step_count += 1
        
        # Check termination conditions:
        # Cart leaves range [-2.4, 2.4] or pole falls beyond 12 degrees (~0.209 rad)
        cart_pos = self.data.qpos[0]
        pole_angle = self.data.qpos[1]
        
        terminated = bool(
            abs(cart_pos) > 2.4
            or abs(pole_angle) > 0.209
            or np.isnan(self.data.qpos).any()
        )
        
        # Standard step-alive reward
        reward = 1.0 if not terminated else 0.0
        
        obs = self._get_obs()
        truncated = self._step_count >= self._max_steps
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
