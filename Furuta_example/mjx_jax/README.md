# MuJoCo MJX (JAX/GPU) Furuta Pendulum Balancing Example

This folder contains a GPU-accelerated training example using **MuJoCo MJX** (MuJoCo's JAX-native pipeline) and **Brax**'s PPO training framework. 

Training runs entirely on the GPU via XLA, compiling the physics engine and RL algorithm into a single fused GPU kernel. This speeds up training by 100x compared to CPU-based training, completing the balance task in under a minute.

## How to Run

1. **Create and activate the MJX virtual environment** (from the repository root):
   ```bash
   python -m venv mjx_env
   source mjx_env/bin/activate
   pip install -r Furuta_example/requirements.txt
   # Make sure to install the correct JAX GPU package matching your CUDA/ROCm drivers, e.g.:
   # pip install --upgrade "jax[cuda12]"
   ```

2. **Run the training script**:
   ```bash
   cd Furuta_example/mjx_jax
   python train_mjx.py
   ```
   This compiles and trains the PPO agent over millions of steps in parallel, then saves the trained network weights to `furuta_mjx_weights.pkl`.

3. **Evaluate the agent in the viewer** (using the unified evaluator in the parent directory):
   ```bash
   cd ..
   python evaluate_unified.py --model-type mjx --model-path mjx_jax/furuta_mjx_weights.pkl
   ```
   This loads the JAX model weights, runs the policy using CPU MuJoCo, and launches the interactive passive viewer.

