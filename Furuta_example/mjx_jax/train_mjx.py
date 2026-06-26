import os
import pickle
import jax
import jax.numpy as jp
from brax import envs
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

class FurutaMJX(PipelineEnv):
    """
    Standard Cartpole environment on MuJoCo MJX.
    Keeps the class name consistent for easy user commands.
    """
    def __init__(self, **kwargs):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cartpole.xml"))
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")
            
        sys = mjcf.load(xml_path)
        # 2 substeps of 0.01s = 0.02s per step
        super().__init__(sys=sys, backend='mjx', n_frames=2, **kwargs)

    def reset(self, rng):
        rng, rng_cart, rng_pole = jax.random.split(rng, 3)
        
        qpos = jp.zeros(self.sys.nq)
        qvel = jp.zeros(self.sys.nv)
        
        # Perturb slightly around upright center
        qpos = qpos.at[0].set(jax.random.uniform(rng_cart, minval=-0.05, maxval=0.05))
        qpos = qpos.at[1].set(jax.random.uniform(rng_pole, minval=-0.05, maxval=0.05))
        
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        
        reward, done = jp.ones(()), jp.zeros(())
        metrics = {}
        
        return envs.State(pipeline_state, obs, reward, done, metrics)

    def step(self, state, action):
        action = jp.clip(action, -1.0, 1.0)
        
        # Step physics using MJX
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        
        obs = self._get_obs(pipeline_state)
        qpos = pipeline_state.qpos
        
        # Termination conditions
        cart_pos = qpos[0]
        pole_angle = qpos[1]
        
        unstable = (jp.abs(cart_pos) > 2.4) | (jp.abs(pole_angle) > 0.209) | jp.isnan(qpos).any()
        
        reward = jp.where(unstable, 0.0, 1.0)
        done = jp.where(unstable, 1.0, 0.0)
        
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state):
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        return jp.array([
            qpos[0],  # cart position
            qvel[0],  # cart velocity
            qpos[1],  # pole angle
            qvel[1]   # pole velocity
        ])

def main():
    print("🚀 Registering FurutaMJX (Cartpole) environment in Brax registry...")
    envs.register_environment('furuta_mjx', FurutaMJX)
    
    env = envs.get_environment('furuta_mjx')

    print("🏋️ Launching JAX-compiled PPO training (GPU accelerated)...")
    # Train over 1,000,000 steps with 256 parallel environments (runs in ~15s on GPU!)
    def progress_fn(num_steps, metrics):
        print(f"Step: {num_steps:7d} | Reward: {metrics['eval/episode_reward']:.3f}")

    train_fn = ppo.train(
        env,
        num_timesteps=1_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_envs=256,
        learning_rate=1e-3,
        entropy_cost=1e-2,
        discounting=0.99,
        seed=0,
        progress_fn=progress_fn
    )

    make_inference_fn, params, _ = train_fn

    model_path = "furuta_mjx_weights.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
        
    print(f"✅ Training completed successfully! Model weights saved to '{model_path}'")

if __name__ == "__main__":
    main()
