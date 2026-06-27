# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a whole_body_control PPO policy as a LIFT3-matched baseline.

Differences vs scripts/rsl_rl/train.py:
  * motion is loaded from a LOCAL --motion_file path (mirrors the bm_wbt
    collector), NOT from a wandb registry, so it uses the exact same
    motion_fight1_subject2_cut2_mujoco.npz as the GPU0-5 experiments;
  * env is reward/physics-matched to the GPU0-5 SAC: undesired_contacts reward
    DISABLED (the SAC uses brax_compute_reward, which has no contact penalty),
    and ALL domain-randomization events disabled;
  * num_envs=1000, num_steps_per_env=20, save_interval=50;
  * adaptive motion-frame sampling stays ON (env default);
  * logs Metrics/avg_total_reward + Metrics/avg_episode_length over each
    1000-step window (see lift_metrics.py).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a LIFT-matched whole_body_control PPO agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-Wo-State-Estimation-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="Local path to the motion npz (no wandb registry).")
parser.add_argument("--window_steps", type=int, default=1000, help="Steps per avg_total_reward window.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.tracking.mdp as mdp  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402

# LIFT-matched metrics wrapper + runner (sibling module)
from lift_metrics import LiftMetricsOnPolicyRunner, LiftMetricsVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # ---- LIFT3-matched PPO knobs (user spec) ----
    agent_cfg.num_steps_per_env = 20
    # Policy saving DISABLED (huge interval): the held-out eval runs in-process on
    # the live policy, so no disk checkpoints are needed (frees disk).
    agent_cfg.save_interval = 100000000
    if not agent_cfg.experiment_name:
        agent_cfg.experiment_name = "g1_flat"
    agent_cfg.experiment_name = agent_cfg.experiment_name + "_lift_match"

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- LIFT3 reward match: GPU0-5 SAC trains on brax_compute_reward, which
    # has the SAME tracking + action_rate + joint_limit terms but NO contact
    # penalty -> disable undesired_contacts so the PPO reward matches exactly.
    env_cfg.rewards.undesired_contacts = None

    # ---- No domain randomization (match the collector / GPU0-5 setup) ----
    env_cfg.events.physics_material = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    # NOTE: keep the add_joint_default_pos event but turn OFF its randomization.
    # Its function (randomize_joint_default_pos) unconditionally sets the custom
    # data.default_joint_pos_nominal attribute (read by the ONNX exporter at
    # save time); only the +-0.01 rad noise is gated on pos_distribution_params.
    # Nulling that param => no domain randomization, but the attribute is still
    # created (else runner.save() at iter 0 crashes in the exporter).
    env_cfg.events.add_joint_default_pos.params["pos_distribution_params"] = None

    # ---- Observation noise OFF + reset-state noise OFF (match GPU0-4 collected
    # data, which the collector produces with enable_corruption=False and the
    # MotionCommand pose/velocity/joint reset ranges zeroed). ----
    env_cfg.observations.policy.enable_corruption = False
    _zero6 = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
              "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.commands.motion.pose_range = dict(_zero6)
    env_cfg.commands.motion.velocity_range = dict(_zero6)
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    # ---- Terminations aligned to the GPU0-4 brax done (the ACTUAL conditions;
    # the height/lin-vel/ang-vel/joint/motor-vel block in brax is_done is dead
    # code, commented out). brax done = anchor pos-z |dz|>0.35 OR anchor
    # gravity-z |d|>0.8 OR terminate-body pos-z |dz|>0.25 (ankles+wrists) OR
    # motion-end. PPO already has z-only anchor_pos / gravity-z anchor_ori (0.8)
    # / z-only ee_body_pos (0.25, 4 EE); fix anchor_pos threshold and add the
    # missing motion-end termination (otherwise PPO loops the clip).
    env_cfg.terminations.anchor_pos.params["threshold"] = 0.35
    env_cfg.terminations.motion_end = DoneTerm(
        func=mdp.motion_reached_end, params={"command_name": "motion"}
    )

    # ---- Motion from a local npz path (mirror the bm_wbt collector; no registry).
    # Adaptive motion-frame sampling stays ON (commands.motion.start_frame is None
    # by default).
    motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    env_cfg.commands.motion.motion_file = str(motion_path)
    print(f"[INFO] LIFT-matched PPO | motion={motion_path}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs} num_steps_per_env={agent_cfg.num_steps_per_env} "
          f"save_interval={agent_cfg.save_interval} window_steps={args_cli.window_steps}")
    print("[INFO] undesired_contacts=OFF | domain_randomization=OFF | adaptive_sampling=ON | terminations=default")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl (+ LIFT windowed metrics)
    env = LiftMetricsVecEnvWrapper(env, window_steps=args_cli.window_steps)

    # create runner from rsl-rl (no wandb registry artifact to link)
    runner = LiftMetricsOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
