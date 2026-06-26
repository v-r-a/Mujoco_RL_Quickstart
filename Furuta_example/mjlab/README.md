# mjlab (PyTorch + rsl_rl) Furuta Pendulum Balancing Example

This folder illustrates how to define tasks and train agents using the **`mjlab`** framework (which uses modular, manager-based configurations similar to Isaac Lab/Orbit) and the **`rsl_rl`** on-policy training library.

In `mjlab`, environments are configured declaratively using `ManagerBasedRlEnvCfg`. Features like observations, rewards, events, and actions are added as config terms managed by respective manager classes.

## How to Run

1. **Create and activate the mjlab virtual environment** (from the repository root):
   ```bash
   python -m venv mjlab_env
   source mjlab_env/bin/activate
   pip install -r Furuta_example/mjlab/requirements.txt
   ```

2. **Register the Task and Run Training**:
   Add the task configuration to your local `mjlab` workspace registry (usually under `src/mjlab/tasks`), or run the training runner directly:
   ```bash
   cd Furuta_example/mjlab
   python register_task.py
   ```
   This will initialize the manager-based environment and launch the `rsl_rl` on-policy PPO training runner.

3. **Evaluate the agent in the viewer** (using the unified evaluator in the parent directory):
   ```bash
   cd ..
   python evaluate_unified.py --model-type mjlab --model-path mjlab/logs/rsl_rl/model_20.pt
   ```
   This loads the PyTorch model checkpoint, runs the policy using CPU MuJoCo, and launches the interactive passive viewer.

