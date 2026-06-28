from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    out = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    # Suppressed during the held-out eval (eval_mode): the eval judges failure itself
    # AFTER env.step using the refreshed body/anchor poses (matching the LIFT brax-flags
    # eval), so the env must NOT auto-reset on the reset-step stale pose (which would
    # spuriously fail step 1 -> ep_len=1). Training (eval_mode=False) is unaffected.
    if getattr(command, "eval_mode", False):
        return torch.zeros_like(out)
    return out


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    out = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    if getattr(command, "eval_mode", False):  # suppressed during eval (see bad_anchor_pos_z_only)
        return torch.zeros_like(out)
    return out


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    out = torch.any(error > threshold, dim=-1)
    if getattr(command, "eval_mode", False):  # suppressed during eval (see bad_anchor_pos_z_only)
        return torch.zeros_like(out)
    return out


def motion_reached_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the motion clip reaches its last frame (matches the brax
    GPU0-4 done: ``ref_index >= time_step_total - 1``). Checked before the command
    manager advances/wraps time_steps (termination runs before command.compute),
    so it fires on the final frame instead of letting the MotionCommand loop.

    During the LIFT held-out eval (``command.eval_mode``) this never fires, so a
    clip that reaches its end loops via adaptive resample WITHOUT counting as a
    ``done`` (matches the eval spec)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reached = command.time_steps >= (command.motion.time_step_total - 1)
    if getattr(command, "eval_mode", False):
        return torch.zeros_like(reached)
    return reached
