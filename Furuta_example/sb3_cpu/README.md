# Stable-Baselines3 (CPU) Furuta Pendulum Balancing Example

This folder contains a standard PyTorch + Stable-Baselines3 (SB3) CPU training example. It uses Gymnasium to define a custom environment that wraps MuJoCo Python bindings.

## How to Run

1. **Create and activate the SB3 virtual environment** (from the repository root):
   ```bash
   python -m venv sb3_env
   source sb3_env/bin/activate
   pip install -r Furuta_example/sb3_cpu/requirements.txt
   ```

2. **Run the training script**:
   ```bash
   cd Furuta_example/sb3_cpu
   python train.py
   ```
   This will train a PPO agent for 300,000 timesteps and save the model as `furuta_ppo_model.zip`.

3. **Evaluate and visualize the trained agent** (using the unified evaluator in the parent directory):
   ```bash
   cd ..
   python evaluate_unified.py --model-type sb3 --model-path sb3_cpu/furuta_ppo_model.zip
   ```
   This opens the passive MuJoCo viewer and plays back the trained agent in real-time.

