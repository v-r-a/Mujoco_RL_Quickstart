# MuJoCo RL Quickstart: Comparing SB3, MJX, mjlab

Welcome to the **MuJoCo RL Quickstart**! This repository is designed to help you learn and compare three major reinforcement learning (RL) paradigms for training agents in physical simulations using **MuJoCo**:

1. **`stable-baselines3` (CPU)**: A classic, user-friendly, CPU-bound RL library using standard Gymnasium wrappers.
2. **`MJX` (JAX/GPU)**: MuJoCo's JAX-native physics engine combined with Brax's PPO, compiling the simulator and training loop into unified GPU kernels for 100x speedups.
3. **`mjlab` (PyTorch)**: A modular, manager-based environment configuration framework (similar to Isaac Lab/Orbit) using `rsl_rl` for structured on-policy PPO training.

All three paradigms are applied to the same physical system: the **Furuta Pendulum (Cartpole)**, allowing a direct side-by-side comparison of environment configuration, training performance, and evaluation.

---



---

## 📁 Repository Structure

```text
.
├── Furuta_example/             # Main example directory
│   ├── cartpole.xml            # Shared MuJoCo physics model (MJCF)
│   ├── evaluate_unified.py     # Unified script to run and visualize any trained policy
│   ├── requirements.txt        # Combined Python dependencies for all frameworks
│   │
│   ├── sb3_cpu/                # Stable-Baselines3 CPU implementation
│   │   ├── train.py            # SB3 PPO training script
│   │   └── furuta_ppo_model.zip# Pre-trained SB3 model checkpoint
│   │
│   ├── mjx_jax/                # MuJoCo MJX JAX-GPU implementation
│   │   ├── train_mjx.py        # MJX JAX training script
│   │   └── furuta_mjx_weights.pkl # Pre-trained MJX model weights
│   │
│   └── mjlab/                  # mjlab PyTorch manager-based implementation
│       ├── furuta_task_cfg.py  # Declarative manager environment config
│       ├── register_task.py    # Environment registration and rsl_rl runner
│       └── logs/rsl_rl/        # Local training log outputs
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have a system with:
* Python 3.8+ (recommended: Python 3.10)
* A CUDA-compatible GPU (highly recommended for `MJX` and `mjlab`)

### 2. Environment Setup
To keep dependencies clean, it is recommended to create separate virtual environments for each framework, or install dependencies as needed:

* **Stable-Baselines3**:
  ```bash
  python -m venv sb3_env
  source sb3_env/bin/activate
  pip install -r Furuta_example/requirements.txt
  deactivate
  ```

* **MJX (JAX/GPU)**:
  ```bash
  python -m venv mjx_env
  source mjx_env/bin/activate
  pip install -r Furuta_example/requirements.txt
  # Make sure to install the JAX version matching your CUDA drivers:
  # pip install --upgrade "jax[cuda12]"
  deactivate
  ```

* **mjlab**:
  ```bash
  python -m venv mjlab_env
  source mjlab_env/bin/activate
  pip install -r Furuta_example/requirements.txt
  # Note: mjlab is a custom repository framework. You will need to install it in this environment
  # (e.g., in editable mode: pip install -e /path/to/mjlab_source)
  deactivate
  ```

---

## 🏋️ Training & Evaluating Agents

Follow the detailed instructions in each folder's sub-README to train your agents:
* Read [sb3_cpu/README.md](file:///home/nandhith/Python/rl_muj_mjx_mjlab/Furuta_example/sb3_cpu/README.md)
* Read [mjx_jax/README.md](file:///home/nandhith/Python/rl_muj_mjx_mjlab/Furuta_example/mjx_jax/README.md)
* Read [mjlab/README.md](file:///home/nandhith/Python/rl_muj_mjx_mjlab/Furuta_example/mjlab/README.md)

### 📺 Unified Evaluation
Once you have trained models (or want to test the pre-trained weights), use the **`evaluate_unified.py`** script. This loads the model weights and runs them inside the standard CPU-based MuJoCo interactive viewer.

Run the script from the `Furuta_example` directory using the appropriate environment:

* **Evaluate Stable-Baselines3**:
  ```bash
  source sb3_env/bin/activate
  cd Furuta_example
  python evaluate_unified.py --model-type sb3 --model-path sb3_cpu/furuta_ppo_model.zip
  ```

* **Evaluate MJX JAX**:
  ```bash
  source mjx_env/bin/activate
  cd Furuta_example
  python evaluate_unified.py --model-type mjx --model-path mjx_jax/furuta_mjx_weights.pkl
  ```

* **Evaluate mjlab (rsl_rl)**:
  ```bash
  source mjlab_env/bin/activate
  cd Furuta_example
  python evaluate_unified.py --model-type mjlab --model-path mjlab/logs/rsl_rl/model_20.pt
  ```
