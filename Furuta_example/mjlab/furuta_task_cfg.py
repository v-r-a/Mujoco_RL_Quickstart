import os
from dataclasses import dataclass
import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.envs.mdp.terminations import time_out
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlMotorActuatorCfg

# Define standard cartpole reward function
def cartpole_reward(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cart_pos = robot.data.joint_pos[:, 0]
    pole_angle = robot.data.joint_pos[:, 1]
    
    # Check if failed (cart out of bounds or pole fallen over)
    failed = (torch.abs(cart_pos) > 2.4) | (torch.abs(pole_angle) > 0.209)
    
    # 1.0 for staying upright, 0.0 if failed
    return torch.where(failed, torch.zeros_like(cart_pos), torch.ones_like(cart_pos))

# Define custom termination condition for failure
def cartpole_failure_term(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cart_pos = robot.data.joint_pos[:, 0]
    pole_angle = robot.data.joint_pos[:, 1]
    
    # Check if failed (cart out of bounds or pole fallen over)
    failed = (torch.abs(cart_pos) > 2.4) | (torch.abs(pole_angle) > 0.209)
    return failed

@dataclass
class FurutaEnvCfg(ManagerBasedRlEnvCfg):
    """Full environment configuration in mjlab style for Cartpole."""
    
    # Simulation setup: 0.01s timestep, with decimation=2 -> 0.02s (50 Hz) control steps
    sim: SimulationCfg = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.01
        )
    )
    decimation: int = 2
    
    # Scene contains the robot model
    scene: SceneCfg = SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        entities={
            "robot": EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_file(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../cartpole.xml"))
                ),
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlMotorActuatorCfg(joint_names_expr=("slider",)),
                    )
                )
            )
        }
    )
    
    # Action spaces are mapped to the cart slide actuator
    actions: dict = None
    
    # Observations dictionary matching the env config schema
    observations: dict = None
    
    # Rewards dictionary mapping terms
    rewards: dict = None

    # Terminations dictionary
    terminations: dict = None

    # Episode limit (10 seconds)
    episode_length_s: float = 10.0

    def __post_init__(self):
        self.actions = {
            "joint_torque": JointEffortActionCfg(
                asset_name="robot",
                actuator_names=("slider",),
                scale=1.0
            )
        }
        
        self.observations = {
            "policy": ObservationGroupCfg(
                terms={
                    "cart_pos": ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_pos[:, 0].unsqueeze(-1)),
                    "cart_vel": ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_vel[:, 0].unsqueeze(-1)),
                    "pole_angle": ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_pos[:, 1].unsqueeze(-1)),
                    "pole_vel": ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_vel[:, 1].unsqueeze(-1)),
                },
                concatenate_terms=True,
                # Enable observation normalization
            )
        }
        
        self.rewards = {
            "balance": RewardTermCfg(
                func=cartpole_reward,
                weight=1.0
            )
        }

        self.terminations = {
            "time_out": TerminationTermCfg(
                func=time_out,
                time_out=True
            ),
            "cartpole_failure": TerminationTermCfg(
                func=cartpole_failure_term
            )
        }
