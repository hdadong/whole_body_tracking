"""LIFT-comparison custom metrics for the whole_body_tracking PPO baseline.

Two layers on top of rsl_rl's defaults, matched to how the LIFT3 fight1 baselines
are summarised (shared ``fight1_baselines`` wandb project, identical metric names):

* ``Metrics/avg_total_reward`` / ``Metrics/avg_episode_length`` - cheap windowed
  training-rollout stats over each ``window_steps`` (default 1000) env steps
  (the original behaviour; kept as secondary diagnostics).

* ``eval/*`` - a held-out 50-env evaluation (see ``lift_eval.run_isaaclab_eval``)
  run at the phased cadence: every 20k env-steps while total < 2e5, then every 1M
  env-steps. These are THE cross-baseline comparison metrics, logged against
  ``eval/env_steps`` (total real *training* env-steps) so the SAC / PPO / FastSAC
  runs line up on a common x-axis.
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from lift_eval import run_isaaclab_eval

# eval cadence (total training env-steps): every EVAL_INTERVAL_EARLY while below
# EVAL_PHASE_SWITCH, then every EVAL_INTERVAL_LATE.
EVAL_PHASE_SWITCH = 200_000
EVAL_INTERVAL_EARLY = 20_000
EVAL_INTERVAL_LATE = 1_000_000


class LiftMetricsVecEnvWrapper(RslRlVecEnvWrapper):
    """rsl_rl env wrapper that accumulates windowed return + ended-episode lengths.

    Windowed accumulation is paused (``eval_active``) while the held-out eval steps
    the same env, so eval rollouts do not pollute the training-window metrics."""

    def __init__(self, env, window_steps: int = 1000):
        super().__init__(env)
        self.window_steps = int(window_steps)
        self._win_return = None   # [num_envs] sum of per-step rewards this window
        self._cur_ep_len = None   # [num_envs] steps since each env's last reset
        self._win_ep_lens = []    # lengths of episodes that ended this window
        self._win_step = 0        # env steps elapsed in the current window
        self.pending_metrics = None  # closed-window values, flushed by the runner
        self.eval_active = False  # when True, skip windowed accumulation

    def _lazy_init(self, rew):
        if self._win_return is None:
            self._win_return = torch.zeros_like(rew)
            self._cur_ep_len = torch.zeros_like(rew)

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        if self.eval_active:
            return obs, rew, dones, extras
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
    """MotionOnPolicyRunner that flushes window metrics each ``log()`` and runs the
    held-out 50-env eval at the phased cadence inside an overridden ``learn()``."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None,
                 eval_num_envs: int = 50, eval_max_steps: int = 600):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device, registry_name=registry_name)
        self.lift_eval_num_envs = int(eval_num_envs)
        self.lift_eval_max_steps = int(eval_max_steps)
        self._next_eval_env_steps = EVAL_INTERVAL_EARLY  # first eval at 20k env-steps
        self._eval_defined_metric = False

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

    # ------------------------------------------------------------------ eval ----
    def _schedule_next_eval(self, env_steps: int):
        interval = EVAL_INTERVAL_EARLY if env_steps < EVAL_PHASE_SWITCH else EVAL_INTERVAL_LATE
        self._next_eval_env_steps = env_steps + interval

    def _define_eval_metrics(self):
        if self._eval_defined_metric or self.disable_logs or self.log_dir is None:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.define_metric("eval/env_steps")
                wandb.define_metric("eval/*", step_metric="eval/env_steps")
        except Exception:
            pass
        self._eval_defined_metric = True

    def _run_lift_eval(self, env_steps: int, eval_step: int):
        # ``eval_step`` is the wandb/tb x-step for this eval. The initial
        # (env_steps=0) eval uses step 0; in-loop eval after iteration ``it`` uses
        # step it+1, so consecutive evals never share a step (else wandb would
        # overwrite the env_steps=0 baseline). The true comparison x-axis is
        # ``eval/env_steps`` (set via wandb define_metric).
        policy = self.get_inference_policy(device=self.env.device)  # deterministic; sets eval_mode
        metrics = run_isaaclab_eval(
            self.env, policy, num_eval_envs=self.lift_eval_num_envs, max_steps=self.lift_eval_max_steps
        )
        metrics["eval/env_steps"] = float(env_steps)
        if self.writer is not None and not self.disable_logs:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, eval_step)
        print(
            f"[lift-eval] env_steps={env_steps} "
            f"avg_total_reward={metrics['eval/avg_total_reward']:.4f} "
            f"avg_episode_length={metrics['eval/avg_episode_length']:.2f} "
            f"ep_len[min={metrics.get('eval/ep_len_min', float('nan')):.0f},"
            f"max={metrics.get('eval/ep_len_max', float('nan')):.0f}] "
            f"err_body_pos={metrics.get('eval/err_body_pos', float('nan')):.4f} "
            f"(episodes={int(metrics['eval/num_episodes'])}, finished={int(metrics['eval/num_finished'])})",
            flush=True,
        )

    # rsl_rl 2.3.1 OnPolicyRunner.learn() faithfully reproduced, with the held-out
    # eval inserted after each iteration's log()+save and a clean obs refresh.
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
        self._define_eval_metrics()

        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # ---- Aligned baseline eval at env_steps=0 (initial, un-updated policy) ----
        # All three baselines log this point so their curves share an aligned start.
        if not self.disable_logs:
            self._run_lift_eval(0, eval_step=0)

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    self.alg.process_env_step(rewards, dones, infos)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            # ---- LIFT held-out 50-env eval at the phased cadence ----
            env_steps = (it + 1) * self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
            if not self.disable_logs and env_steps >= self._next_eval_env_steps:
                self._run_lift_eval(env_steps, eval_step=it + 1)
                self._schedule_next_eval(env_steps)
                # eval perturbed the sim -> refresh rollout state (fresh, desynced episodes)
                if init_at_random_ep_len:
                    self.env.episode_length_buf = torch.randint_like(
                        self.env.episode_length_buf, high=int(self.env.max_episode_length)
                    )
                obs, extras = self.env.get_observations()
                privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
                obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                obs = self.obs_normalizer(obs)
                if self.privileged_obs_type is not None:
                    privileged_obs = self.privileged_obs_normalizer(privileged_obs)
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                self.train_mode()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
