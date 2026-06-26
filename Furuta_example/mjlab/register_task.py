import os
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.runners import OnPolicyRunner

from furuta_task_cfg import FurutaEnvCfg

def play(env_cfg, rl_cfg, preprocess_rl_cfg):
    # Set num_envs to 1 for playback visualization
    env_cfg.scene.num_envs = 1
    
    # Instantiate environment
    print("🤖 Creating Manager-Based RL Environment for playback...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(env)
    
    # Initialize the training runner
    log_dir = os.path.abspath("./logs/rsl_rl")
    
    # Find latest checkpoint
    import glob
    checkpoint_dir = os.path.join(log_dir, "furuta_balance", "*")
    checkpoint_dirs = glob.glob(checkpoint_dir)
    if not checkpoint_dirs:
        print("❌ No checkpoints found in logs/rsl_rl/furuta_balance/")
        return
    # Get latest folder
    latest_run_dir = max(checkpoint_dirs, key=os.path.getctime)
    checkpoint_files = glob.glob(os.path.join(latest_run_dir, "model_*.pt"))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {latest_run_dir}")
        return
    # Find latest model file by modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"🔍 Found latest checkpoint: {latest_checkpoint}")
    
    from dataclasses import asdict
    runner = OnPolicyRunner(
        env=env,
        train_cfg=preprocess_rl_cfg(asdict(rl_cfg)),
        log_dir=log_dir,
        device=env.device
    )
    runner.load(latest_checkpoint, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    
    # Choose viewer based on environment display
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    
    if has_display:
        print("🖥️ Starting Native MuJoCo Viewer...")
        from mjlab.viewer import NativeMujocoViewer
        NativeMujocoViewer(env, policy).run()
    else:
        print("🌐 No local display found. Starting Viser Web-based Playback Viewer...")
        from mjlab.viewer import ViserPlayViewer
        ViserPlayViewer(env, policy).run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Evaluate the latest trained model")
    args = parser.parse_args()

    # 1. Instantiate env configuration
    env_cfg = FurutaEnvCfg()
    
    # 2. Configure the rsl_rl training parameters
    from mjlab.rl.config import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    rl_cfg = RslRlOnPolicyRunnerCfg(
        seed=42,
        max_iterations=150,  # Number of policy updates
        num_steps_per_env=500,  # Episode length
        save_interval=20,
        experiment_name="furuta_balance",
        run_name="mjlab_ppo",
        logger="tensorboard",
        # PPO hyperparameters
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=(128, 128),
            critic_hidden_dims=(128, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",  # adaptive learning rate scheduling
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )

    # 3. Register the task under a custom task ID
    task_id = "Mjlab-Balance-Furuta"
    print(f"📝 Registering task: {task_id}")
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=rl_cfg
    )

    def preprocess_rl_cfg(rl_cfg_dict):
        import copy
        cfg = copy.deepcopy(rl_cfg_dict)
        if "policy" in cfg:
            policy_cfg = cfg.pop("policy")
            cfg["actor"] = {
                "class_name": "rsl_rl.models:MLPModel",
                "hidden_dims": policy_cfg.get("actor_hidden_dims", (128, 128, 128)),
                "activation": policy_cfg.get("activation", "elu"),
                "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
                "distribution_cfg": {
                    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                    "init_std": policy_cfg.get("init_noise_std", 1.0),
                    "std_type": policy_cfg.get("noise_std_type", "scalar"),
                }
            }
            cfg["critic"] = {
                "class_name": "rsl_rl.models:MLPModel",
                "hidden_dims": policy_cfg.get("critic_hidden_dims", (128, 128, 128)),
                "activation": policy_cfg.get("activation", "elu"),
                "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
            }
        cfg["obs_groups"] = {
            "actor": ["policy"],
            "critic": ["policy"],
        }
        if "algorithm" in cfg:
            if "rnd_cfg" not in cfg["algorithm"]:
                cfg["algorithm"]["rnd_cfg"] = None
        return cfg

    if args.play:
        play(env_cfg, rl_cfg, preprocess_rl_cfg)
        return

    # 4. Instantiate the environment
    print("🤖 Creating Manager-Based RL Environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Wrap it to be compatible with rsl_rl API
    env = RslRlVecEnvWrapper(env)

    # 5. Initialize the training runner
    print("🏋️ Launching rsl_rl on-policy PPO training runner...")
    log_dir = os.path.abspath("./logs/rsl_rl")
    from dataclasses import asdict

    runner = OnPolicyRunner(
        env=env,
        train_cfg=preprocess_rl_cfg(asdict(rl_cfg)),
        log_dir=log_dir,
        device=env.device
    )

    # 6. Start training
    runner.learn(
        num_learning_iterations=rl_cfg.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("✅ Training complete! Checkpoints and logs are saved in: ./logs/rsl_rl")

if __name__ == "__main__":
    # Ensure torch is imported inside main or helper
    import torch
    main()
