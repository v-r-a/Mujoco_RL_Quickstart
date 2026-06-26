"""
Unified Evaluation Script for Cartpole
======================================
Loads the Cartpole model in CPU MuJoCo and evaluates a trained model from
either Stable-Baselines3 (sb3), MuJoCo MJX (mjx), or mjlab (rsl_rl).

Usage:
  python evaluate_unified.py --model-type [sb3|mjx|mjlab] --model-path [path_to_model_file]
"""

import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "cartpole.xml"))

def get_obs(data):
    """Generate the standard 4D observation vector [cart_pos, cart_vel, pole_angle, pole_vel]."""
    return np.array([
        data.qpos[0],
        data.qvel[0],
        data.qpos[1],
        data.qvel[1]
    ], dtype=np.float32)

def load_sb3_policy(model_path):
    from stable_baselines3 import PPO
    print(f"📖 Loading Stable-Baselines3 model from: {model_path}")
    model = PPO.load(model_path)
    
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action[0]
        
    return policy

def load_mjx_policy(model_path):
    import pickle
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    
    print(f"📖 Loading JAX MJX weights from: {model_path}")
    with open(model_path, "rb") as f:
        params = pickle.load(f)
        
    # Reconstruct network architecture (4D input, 1D output)
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=4,
        action_size=1,
        preprocess_observations_fn=running_statistics.normalize
    )
    make_inference = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference(params, deterministic=True)
    apply_fn = jax.jit(inference_fn)
    
    def policy(obs):
        rng = jax.random.PRNGKey(0)
        obs_batched = jp.expand_dims(jp.array(obs), axis=0)
        action_batched, _ = apply_fn(obs_batched, rng)
        return np.array(action_batched)[0, 0]
        
    return policy

def load_mjlab_policy(model_path):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    
    print(f"📖 Loading mjlab (rsl_rl) checkpoint from: {model_path}")
    
    # Create dummy observation to construct the model
    dummy_obs_tensor = torch.zeros((1, 4))
    obs_td = TensorDict({"policy": dummy_obs_tensor}, batch_size=[1])
    
    actor = MLPModel(
        obs=obs_td,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        obs_set="actor",
        output_dim=1,
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar"
        }
    )
    
    checkpoint = torch.load(model_path, map_location="cpu")
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    def policy(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_td = TensorDict({"policy": obs_tensor}, batch_size=[1])
        with torch.no_grad():
            action = actor(obs_td).cpu().numpy()[0]
        return action[0]
        
    return policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cartpole trained policy on CPU MuJoCo.")
    parser.add_argument("--model-type", type=str, required=True, choices=["sb3", "mjx", "mjlab"],
                        help="The paradigm used to train the model (sb3, mjx, or mjlab)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model zip, pkl, or pt checkpoint file")
    args = parser.parse_args()

    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML file not found at: {XML_PATH}")

    # Load policy depending on selected framework
    if args.model_type == "sb3":
        policy_fn = load_sb3_policy(args.model_path)
    elif args.model_type == "mjx":
        policy_fn = load_mjx_policy(args.model_path)
    else:
        policy_fn = load_mjlab_policy(args.model_path)

    # Initialize CPU MuJoCo
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    def reset_sim():
        mujoco.mj_resetData(model, data)
        data.qpos[0] = np.random.uniform(-0.05, 0.05)
        data.qpos[1] = np.random.uniform(-0.05, 0.05)
        data.qvel[0] = np.random.uniform(-0.05, 0.05)
        data.qvel[1] = np.random.uniform(-0.05, 0.05)
        mujoco.mj_forward(model, data)

    reset_sim()
    print("📺 Opening MuJoCo passive viewer. Close window to stop.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            obs = get_obs(data)
            action = policy_fn(obs)
            
            # Apply control action directly (unscaled sliding force [-1, 1])
            data.ctrl[0] = np.clip(action, -1.0, 1.0)
            for _ in range(2):
                mujoco.mj_step(model, data)
                
            viewer.sync()
            
            # Reset if cart falls out of bounds or pole falls over
            if abs(data.qpos[0]) > 2.4 or abs(data.qpos[1]) > 0.209:
                reset_sim()

            # Maintain 50 Hz control rate (dt = 0.02s)
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
