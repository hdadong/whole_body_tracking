"""LIFT-comparison custom metrics for the whole_body_tracking PPO baseline.

Adds two wandb metrics, on top of rsl_rl's defaults, matched to the way the
LIFT3 GPU0-5 SAC experiments are summarised:

* ``Metrics/avg_total_reward`` - each environment accumulates its per-step
  reward (sum each step) over a fixed window of ``window_steps`` (default 1000)
  environment steps; once every env has stepped ``window_steps`` times the
  per-env totals are averaged across all (1000) envs and logged, then reset to
  zero and accumulation restarts.
* ``Metrics/avg_episode_length`` - mean length of the episodes that *ended*
  (terminated or timed out) within that same window.

With ``num_steps_per_env=20`` a 1000-step window is exactly 50 PPO iterations,
so a window always closes on an iteration boundary; the closed-window values are
stashed on the env wrapper and flushed to the writer by the runner's ``log()``
(which fires once per iteration), keeping a single, monotonic wandb step axis.
"""

from __future__ import annotations

import statistics

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


class LiftMetricsVecEnvWrapper(RslRlVecEnvWrapper):
    """rsl_rl env wrapper that accumulates windowed return + ended-episode lengths."""

    def __init__(self, env, window_steps: int = 1000):
        super().__init__(env)
        self.window_steps = int(window_steps)
        self._win_return = None   # [num_envs] sum of per-step rewards this window
        self._cur_ep_len = None   # [num_envs] steps since each env's last reset
        self._win_ep_lens = []    # lengths of episodes that ended this window
        self._win_step = 0        # env steps elapsed in the current window
        self.pending_metrics = None  # closed-window values, flushed by the runner

    def _lazy_init(self, rew):
        if self._win_return is None:
            self._win_return = torch.zeros_like(rew)
            self._cur_ep_len = torch.zeros_like(rew)

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        self._lazy_init(rew)

        # per-step accumulation (all envs step in lockstep)
        self._win_return += rew
        self._cur_ep_len += 1

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self._win_ep_lens.extend(self._cur_ep_len[done_ids].detach().cpu().numpy().tolist())
            self._cur_ep_len[done_ids] = 0.0

        self._win_step += 1
        if self._win_step >= self.window_steps:
            self.pending_metrics = {
                "Metrics/avg_total_reward": float(self._win_return.mean().item()),
                "Metrics/avg_episode_length": (
                    float(statistics.mean(self._win_ep_lens)) if self._win_ep_lens else 0.0
                ),
                "Metrics/num_ended_episodes": float(len(self._win_ep_lens)),
            }
            self._win_return.zero_()
            self._win_ep_lens = []
            self._win_step = 0

        return obs, rew, dones, extras


class LiftMetricsOnPolicyRunner(MotionOnPolicyRunner):
    """MotionOnPolicyRunner that flushes the wrapper's window metrics each log()."""

    def log(self, locs, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        pending = getattr(self.env, "pending_metrics", None)
        if pending is not None and self.log_dir is not None and not self.disable_logs:
            for key, value in pending.items():
                self.writer.add_scalar(key, value, locs["it"])
            print(
                f"[lift-metrics] iter={locs['it']} "
                f"avg_total_reward={pending['Metrics/avg_total_reward']:.4f} "
                f"avg_episode_length={pending['Metrics/avg_episode_length']:.2f} "
                f"(ended_episodes={int(pending['Metrics/num_ended_episodes'])}, window=1000 steps)",
                flush=True,
            )
            self.env.pending_metrics = None
