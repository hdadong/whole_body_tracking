# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the LIFT3 FastSAC algorithm on the IsaacLab whole_body_tracking env.

Ports holosoma's FastSAC (C51 distributional critic + Q-ensemble + UTD + obs
normalization + alpha autotune) to run on the EXACT same IsaacLab G1 tracking env
as the PPO baseline (train_lift_match.py): identical motion / physics / reset /
termination / reward / eval. The only env difference is the action term scale
(0.25) so the actor's tanh action space maps back to FastSAC's per-joint range.

See lift_sac_runner.LiftSACRunner for the training loop and fast_sac_net /
fast_sac_buffer for the ported networks / replay buffer.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a LIFT-matched FastSAC agent on IsaacLab whole_body_tracking.")
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-Wo-State-Estimation-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=400000, help="SAC training iterations (1 env-step each).")
parser.add_argument("--motion_file", type=str, required=True, help="Local path to the motion npz (no wandb registry).")
# SAC hyperparameter overrides (default to holosoma FastSACConfig values)
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--tau", type=float, default=0.125)
parser.add_argument("--num_updates", type=int, default=8)
parser.add_argument("--policy_frequency", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--buffer_size", type=int, default=1024)
parser.add_argument("--num_atoms", type=int, default=101)
parser.add_argument("--v_min", type=float, default=-20.0)
parser.add_argument("--v_max", type=float, default=20.0)
parser.add_argument("--actor_lr", type=float, default=3e-4)
parser.add_argument("--critic_lr", type=float, default=3e-4)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime

import torch
import wandb

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.tracking.mdp as mdp  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402

from lift_metrics import LiftMetricsVecEnvWrapper
from lift_sac_runner import LiftSACRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class SACCfg:
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    log_std_max: float = 0.0
    log_std_min: float = -5.0
    use_layer_norm: bool = True
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    num_q_networks: int = 2
    alpha_init: float = 0.001
    target_entropy_ratio: float = 0.0
    use_autotune: bool = True
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    alpha_lr: float = 3e-4
    weight_decay: float = 0.001
    gamma: float = 0.97
    tau: float = 0.125
    batch_size: int = 8192
    buffer_size: int = 1024
    num_steps: int = 1
    num_updates: int = 8
    policy_frequency: int = 4
    learning_starts: int = 10
    max_grad_norm: float = 0.0
    action_term_scale: float = 0.25  # = holosoma control.action_scale
    num_eval_envs: int = 50
    log_interval: int = 100


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ===== env overrides: IDENTICAL to train_lift_match.py (same env as PPO) =====
    env_cfg.rewards.undesired_contacts = None
    env_cfg.events.physics_material = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    env_cfg.events.add_joint_default_pos.params["pos_distribution_params"] = None
    env_cfg.observations.policy.enable_corruption = False
    _zero6 = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
              "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.commands.motion.pose_range = dict(_zero6)
    env_cfg.commands.motion.velocity_range = dict(_zero6)
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.terminations.anchor_pos.params["threshold"] = 0.35
    # motion_end as TIMEOUT (time_out=True) -> truncated, so the SAC bootstraps the
    # clip-end value (matches holosoma's motion_ends is_timeout=True). A clip that
    # reaches its end is NOT a failure; treating it as a real terminated done
    # (bootstrap=0) makes the C51 critic undervalue late-episode states and collapses
    # the policy (entropy->0, ep_len 50->4). bad_tracking stays terminated (bootstrap=0).
    env_cfg.terminations.motion_end = DoneTerm(
        func=mdp.motion_reached_end, params={"command_name": "motion"}, time_out=True
    )

    motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    env_cfg.commands.motion.motion_file = str(motion_path)

    # ===== action term scale = 0.25 (replicate FastSAC action space; see plan A1/A4) =====
    env_cfg.actions.joint_pos.scale = 0.25

    print(f"[INFO] LIFT-matched FastSAC | motion={motion_path}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs} | action_term_scale=0.25 | "
          "undesired_contacts=OFF | DR=OFF | adaptive_sampling=ON")

    # ===== build SAC cfg from argparse =====
    sac_cfg = SACCfg(
        gamma=args_cli.gamma, tau=args_cli.tau, num_updates=args_cli.num_updates,
        policy_frequency=args_cli.policy_frequency, batch_size=args_cli.batch_size,
        buffer_size=args_cli.buffer_size, num_atoms=args_cli.num_atoms,
        v_min=args_cli.v_min, v_max=args_cli.v_max, actor_lr=args_cli.actor_lr, critic_lr=args_cli.critic_lr,
    )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl_sac", "g1_flat_fastsac"))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # ===== wandb (SAC runner logs directly; mirror PPO's fight1_baselines project) =====
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "fight1_baselines"),
            name=os.environ.get("WANDB_RUN_NAME", "fight1_fastsac_isaaclab"),
            entity=os.environ.get("WANDB_ENTITY", "bigeasthuang"),
            dir=log_dir,
            config={**vars(args_cli), **sac_cfg.__dict__},
            settings=wandb.Settings(_disable_stats=True),
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = LiftMetricsVecEnvWrapper(env, window_steps=1000)

    runner = LiftSACRunner(env, sac_cfg, log_dir=log_dir, device=str(env_cfg.sim.device))
    runner.learn(num_iterations=args_cli.max_iterations)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
