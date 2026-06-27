"""Unified 50-env held-out evaluation for the LIFT3 fight1 baselines.

This module runs a held-out evaluation on the *same* whole_body_control IsaacLab
env used for training (one sim context per process, so eval is a discrete phase
that re-uses the training env, not a second env instance):

  * ``num_eval_envs`` (default 50) envs each start on a DIFFERENT motion frame
    (linspace over the clip);
  * the ``motion_reached_end`` termination is suppressed (``command.eval_mode``),
    so reaching the clip end loops via adaptive resample WITHOUT a ``done`` -- only
    a failure termination or the 10 s time-out ends an eval episode;
  * each eval env contributes exactly ONE episode -> 50 episodes total.

It reports, averaged over the 50 episodes, with names identical across the SAC /
PPO / FastSAC baselines so they line up in the shared ``fight1_baselines`` wandb
project:

  eval/avg_total_reward, eval/avg_episode_length,
  eval/reward_<term>   for the 8 reward sub-terms (episode-summed, sum ~= total),
  eval/err_<name>      for the 10 tracking errors  (per-step mean over the episode).

The reward/eval rollout is policy-agnostic: pass ``act_fn(obs)->actions`` (the
PPO trainer passes its deterministic inference policy; the SAC collector passes
its JAX policy wrapper).
"""

from __future__ import annotations

import torch

# Canonical tracking-error names (== MotionCommand.metrics keys without "error_").
ERR_NAMES = (
    "anchor_pos",
    "anchor_rot",
    "anchor_lin_vel",
    "anchor_ang_vel",
    "body_pos",
    "body_rot",
    "body_lin_vel",
    "body_ang_vel",
    "joint_pos",
    "joint_vel",
)


def canonical_reward_name(term: str) -> str | None:
    """Map a framework-specific reward-term name to the canonical baseline name."""
    t = term.lower()
    if "anchor" in t and ("pos" in t or "position" in t):
        return "anchor_pos"
    if "anchor" in t and ("ori" in t or "orientation" in t or "rot" in t):
        return "anchor_ori"
    if "body" in t and "lin" in t and "vel" in t:
        return "body_lin_vel"
    if "body" in t and "ang" in t and "vel" in t:
        return "body_ang_vel"
    if "body" in t and ("pos" in t or "position" in t):
        return "body_pos"
    if "body" in t and ("ori" in t or "orientation" in t or "rot" in t):
        return "body_ori"
    if "action_rate" in t or ("action" in t and "rate" in t):
        return "action_rate"
    if ("joint" in t and "limit" in t) or ("dof" in t and ("limit" in t or "pos" in t)):
        return "dof_pos_limits"
    return None


def run_isaaclab_eval(
    env,
    act_fn,
    command_name: str = "motion",
    num_eval_envs: int = 50,
    max_steps: int = 600,
) -> dict[str, float]:
    """Run the held-out eval on ``env`` and return the canonical ``eval/*`` dict.

    ``env`` is an rsl_rl-style vec-env wrapper exposing ``reset()``/``step()`` and
    ``unwrapped`` (with ``command_manager``, ``reward_manager`` and ``step_dt``).
    ``act_fn`` maps the policy observation tensor to an action tensor.
    """
    unwrapped = env.unwrapped
    device = unwrapped.device
    command = unwrapped.command_manager.get_term(command_name)
    reward_manager = unwrapped.reward_manager
    dt = float(unwrapped.step_dt)

    n_total = unwrapped.num_envs
    n_eval = min(int(num_eval_envs), n_total)
    eval_ids = torch.arange(n_eval, device=device)

    # Eval protocol (06-21 / cross-baseline aligned): each eval env starts on a
    # DISTINCT motion frame (linspace over the clip). eval_mode=True suppresses
    # motion_reached_end, so reaching the clip end loops via adaptive resample (NOT a
    # done). An episode ends only on a real bad-tracking failure (anchor_pos /
    # anchor_ori / ee_body_pos) or the 500-step cap. Each eval env -> one (possibly
    # clip-looping) episode -> 50 episodes total.
    horizon = int(command.motion.time_step_total)
    frames = torch.linspace(0, max(horizon - 1, 0), n_eval, device=device).long()

    term_names = list(reward_manager.active_terms)
    term_to_canon = {i: canonical_reward_name(n) for i, n in enumerate(term_names)}

    ep_total = torch.zeros(n_eval, device=device)
    ep_len = torch.zeros(n_eval, device=device)
    ep_err_count = torch.zeros(n_eval, device=device)
    ep_reward = {c: torch.zeros(n_eval, device=device) for c in set(v for v in term_to_canon.values() if v)}
    ep_err = {name: torch.zeros(n_eval, device=device) for name in ERR_NAMES}
    finished = torch.zeros(n_eval, dtype=torch.bool, device=device)

    LIFT_EVAL_MAX_EP = 500
    with torch.inference_mode():
        # eval_mode=True -> motion_reached_end returns zeros, so the clip end loops
        # via adaptive resample instead of terminating. done = a real bad-tracking
        # failure (anchor/ee); the 500-step cap is the eval time-out horizon.
        command.eval_mode = True
        command._eval_force_frames[:] = -1
        command._eval_force_frames[eval_ids] = frames
        prev_eval_active = getattr(env, "eval_active", None)
        if hasattr(env, "eval_active"):
            env.eval_active = True
        obs, _ = env.reset()

        for _ in range(int(LIFT_EVAL_MAX_EP)):
            actions = act_fn(obs)
            obs, rew, dones, _ = env.step(actions)
            done = dones[:n_eval].to(device) > 0  # bad_tracking (motion_end suppressed)
            live = ~finished
            live_nr = live & (~done)  # exclude the done/reset step from reward+errors
            livef = live.float()
            lnrf = live_nr.float()

            ep_total += rew[:n_eval].to(device) * lnrf
            ep_len += livef  # counts up to & incl. the done step
            step_reward = reward_manager._step_reward  # [num_envs, num_terms], = weighted/dt
            for idx, canon in term_to_canon.items():
                if canon is None:
                    continue
                ep_reward[canon] += step_reward[:n_eval, idx] * dt * lnrf
            for name in ERR_NAMES:
                ep_err[name] += command.metrics["error_" + name][:n_eval] * lnrf
            ep_err_count += lnrf

            my_timeout = ep_len >= float(LIFT_EVAL_MAX_EP)
            finalize = live & (my_timeout | done)
            finished |= finalize
            if bool(finished.all()):
                break

        # Restore training mode.
        command.eval_mode = False
        command._eval_force_frames[:] = -1
        if hasattr(env, "eval_active"):
            env.eval_active = bool(prev_eval_active) if prev_eval_active is not None else False
        env.reset()  # clean reset so the trainer resumes from fresh episodes

    err_denom = ep_err_count.clamp(min=1.0)
    metrics: dict[str, float] = {
        "eval/avg_total_reward": float(ep_total.mean().item()),
        "eval/avg_episode_length": float(ep_len.mean().item()),
        "eval/ep_len_min": float(ep_len.min().item()),
        "eval/ep_len_max": float(ep_len.max().item()),
        "eval/num_episodes": float(n_eval),
        "eval/num_finished": float(int(finished.sum().item())),
    }
    for canon, buf in ep_reward.items():
        metrics[f"eval/reward_{canon}"] = float(buf.mean().item())
    for name in ERR_NAMES:
        metrics[f"eval/err_{name}"] = float((ep_err[name] / err_denom).mean().item())
    return metrics
