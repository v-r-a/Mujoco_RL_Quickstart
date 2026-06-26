import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from furuta_env import FurutaEnv

def main():
    print("🚀 Initializing Stable-Baselines3 training environment (CPU)...")
    
    # Create vectorized environment for training acceleration
    env = make_vec_env(FurutaEnv, n_envs=4)

    # Configure PPO hyper-parameters suitable for continuous control swing-up
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )

    total_timesteps = 300_000
    print(f"🏋️ Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model_name = "furuta_ppo_model"
    model.save(model_name)
    print(f"✅ Training completed! Model saved as '{model_name}.zip'")

if __name__ == "__main__":
    main()
