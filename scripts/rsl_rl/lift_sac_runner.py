"""FastSAC training runner for the IsaacLab whole_body_tracking env.

Ports holosoma's `fast_sac_agent` off-policy loop to rsl_rl's RslRlVecEnvWrapper:
collect 1 env-step/iter -> replay buffer -> UTD update (C51 critic CE + actor +
alpha autotune + soft target) -> periodic held-out eval via `run_isaaclab_eval`.

Faithful to FastSAC: distributional C51 critic, Q-ensemble, UTD (num_updates /
policy_frequency), per-joint action_scale, obs normalization, gamma/tau, the
entropy term folded into the critic projection reward, bootstrap=(trunc|~done).

Differences from holosoma (documented):
  * IsaacLab does NOT expose terminal/final observation (obs_buf is recomputed
    after _reset_idx), so for timeout transitions we store next_obs as the
    bootstrap target (approximation). truncation fraction is logged.
  * single-GPU, no AMP/scaler, no symmetry augmentation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import wandb

from fast_sac_buffer import EmpiricalNormalization, SimpleReplayBuffer
from fast_sac_net import Actor, Critic
from lift_eval import run_isaaclab_eval

# eval cadence (total training env-steps), identical to lift_metrics.py / PPO.
EVAL_PHASE_SWITCH = 200_000
EVAL_INTERVAL_EARLY = 20_000
EVAL_INTERVAL_LATE = 1_000_000


class LiftSACRunner:
    def __init__(self, env, cfg, log_dir, device):
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = device
        self.num_envs = env.num_envs
        self.global_step = 0

        # Probe obs/action dims from a reset.
        obs, extras = env.reset()
        n_obs = obs.shape[1]
        n_critic = extras["observations"]["critic"].shape[1]
        n_act = env.num_actions
        print(f"[sac-runner] n_obs={n_obs} n_critic={n_critic} n_act={n_act} num_envs={self.num_envs}")

        # --- action_scale: replicate FastSAC's action space exactly ---
        # FastSAC: action = tanh(mean) * (max_range/0.25); env then * 0.25 + default
        #          => joint_target = tanh(mean) * max_range + default.
        # IsaacLab: actor outputs tanh(mean)*action_scale; JointPositionAction(scale=0.25,
        #          use_default_offset) gives joint_target = action*0.25 + default.
        # So action_scale = max_range / env_action_scale (env_action_scale = cfg.action_term_scale=0.25).
        robot = env.unwrapped.scene["robot"]
        default = robot.data.default_joint_pos[0].to(device)
        jl = robot.data.joint_pos_limits[0].to(device)  # [n_dof, 2]
        max_range = torch.maximum((jl[:, 0] - default).abs(), (jl[:, 1] - default).abs())
        action_scale = (max_range / cfg.action_term_scale).to(device)
        action_bias = torch.zeros(n_act, device=device)
        print(f"[sac-runner] action_scale (max_range/{cfg.action_term_scale}) min={action_scale.min():.3f} "
              f"max={action_scale.max():.3f}; max_range example[0]={max_range[0]:.4f}")

        # --- networks ---
        self.actor = Actor(
            n_obs=n_obs, n_act=n_act, hidden_dim=cfg.actor_hidden_dim,
            log_std_max=cfg.log_std_max, log_std_min=cfg.log_std_min,
            use_tanh=True, use_layer_norm=cfg.use_layer_norm, device=device,
            action_scale=action_scale, action_bias=action_bias,
        )
        self.qnet = Critic(
            n_obs=n_critic, n_act=n_act, num_atoms=cfg.num_atoms, v_min=cfg.v_min, v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim, use_layer_norm=cfg.use_layer_norm,
            num_q_networks=cfg.num_q_networks, device=device,
        )
        self.qnet_target = Critic(
            n_obs=n_critic, n_act=n_act, num_atoms=cfg.num_atoms, v_min=cfg.v_min, v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim, use_layer_norm=cfg.use_layer_norm,
            num_q_networks=cfg.num_q_networks, device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        self.critic_obs_normalizer = EmpiricalNormalization(shape=n_critic, device=device)

        self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -n_act * cfg.target_entropy_ratio

        self.q_optimizer = torch.optim.AdamW(
            list(self.qnet.parameters()), lr=cfg.critic_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
        self.actor_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()), lr=cfg.actor_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_lr, betas=(0.9, 0.95))

        self.rb = SimpleReplayBuffer(
            n_env=self.num_envs, buffer_size=cfg.buffer_size, n_obs=n_obs, n_act=n_act,
            n_critic_obs=n_critic, n_steps=cfg.num_steps, gamma=cfg.gamma, device=device,
        )

    # ------------------------------------------------------------------
    # SAC updates (ported from fast_sac_agent _update_main / _update_pol,
    # AMP/scaler/all_reduce/symmetry removed).
    # ------------------------------------------------------------------
    def _update_main(self, data):
        cfg = self.cfg
        next_observations = data["next"]["observations"]
        critic_observations = data["critic_observations"]
        next_critic_observations = data["next"]["critic_observations"]
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            next_state_actions, next_state_log_probs = self.actor.get_actions_and_log_probs(next_observations)
            discount = cfg.gamma ** data["next"]["effective_n_steps"]
            target_distributions = self.qnet_target.projection(
                next_critic_observations,
                next_state_actions,
                rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                bootstrap,
                discount,
            )
            target_values = self.qnet_target.get_value(target_distributions)
            target_value_max = target_values.max()
            target_value_min = target_values.min()

        q_outputs = self.qnet(critic_observations, actions)
        critic_log_probs = F.log_softmax(q_outputs, dim=-1)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        if cfg.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=cfg.max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        self.q_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if cfg.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return {
            "buffer_rewards": rewards.mean().detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "qf_loss": qf_loss.detach(),
            "qf_max": target_value_max.detach(),
            "qf_min": target_value_min.detach(),
            "alpha_loss": alpha_loss.detach(),
        }

    def _update_pol(self, data):
        cfg = self.cfg
        critic_observations = data["critic_observations"]
        actions, log_probs = self.actor.get_actions_and_log_probs(data["observations"])
        with torch.no_grad():
            _, _, log_std = self.actor(data["observations"])
            action_std = log_std.exp().mean()
            policy_entropy = -log_probs.mean()

        q_outputs = self.qnet(critic_observations, actions)
        q_probs = F.softmax(q_outputs, dim=-1)
        q_values = self.qnet.get_value(q_probs)
        qf_value = q_values.mean(dim=0)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        self.actor_optimizer.step()
        return {
            "actor_grad_norm": actor_grad_norm.detach(),
            "actor_loss": actor_loss.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
        }

    def _sample_and_prepare_batches(self, batch_size, num_updates):
        large = self.rb.sample(batch_size * num_updates)
        samples_per_update = batch_size * self.num_envs
        # normalize once (normalizer in train() mode -> update=True updates running stats)
        large["observations"] = self.obs_normalizer(large["observations"])
        large["next"]["observations"] = self.obs_normalizer(large["next"]["observations"])
        large["critic_observations"] = self.critic_obs_normalizer(large["critic_observations"])
        large["next"]["critic_observations"] = self.critic_obs_normalizer(large["next"]["critic_observations"])

        batches = []
        for i in range(num_updates):
            s, e = i * samples_per_update, (i + 1) * samples_per_update
            batches.append({
                "observations": large["observations"][s:e],
                "actions": large["actions"][s:e],
                "critic_observations": large["critic_observations"][s:e],
                "next": {
                    "rewards": large["next"]["rewards"][s:e],
                    "dones": large["next"]["dones"][s:e],
                    "truncations": large["next"]["truncations"][s:e],
                    "observations": large["next"]["observations"][s:e],
                    "effective_n_steps": large["next"]["effective_n_steps"][s:e],
                    "critic_observations": large["next"]["critic_observations"][s:e],
                },
            })
        return batches

    # ------------------------------------------------------------------
    def _act_fn(self, policy_obs):
        # deterministic mean action for eval; obs not normalized by caller.
        norm = self.obs_normalizer(policy_obs, update=False)
        return self.actor.explore(norm, deterministic=True)

    def _run_eval(self, env_steps, eval_step):
        was_training = self.obs_normalizer.training
        self.obs_normalizer.eval()
        metrics = run_isaaclab_eval(self.env, self._act_fn, num_eval_envs=self.cfg.num_eval_envs)
        if was_training:
            self.obs_normalizer.train()
        metrics["eval/env_steps"] = float(env_steps)
        if wandb.run is not None:
            wandb.log(metrics, step=eval_step)
        print(
            f"[lift-eval] env_steps={env_steps} "
            f"avg_total_reward={metrics['eval/avg_total_reward']:.4f} "
            f"avg_episode_length={metrics['eval/avg_episode_length']:.2f} "
            f"ep_len[min={metrics['eval/ep_len_min']:.0f},max={metrics['eval/ep_len_max']:.0f}] "
            f"err_body_pos={metrics.get('eval/err_body_pos', float('nan')):.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def learn(self, num_iterations):
        env = self.env
        device = self.device
        cfg = self.cfg

        if wandb.run is not None:
            wandb.define_metric("eval/env_steps")
            wandb.define_metric("eval/*", step_metric="eval/env_steps")

        self._run_eval(env_steps=0, eval_step=0)

        with torch.inference_mode():
            obs, extras = env.reset()
        critic_obs = extras["observations"]["critic"]
        dones = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        next_eval = EVAL_INTERVAL_EARLY

        for it in range(num_iterations):
            self.global_step = it
            # --- collect 1 env-step (inference_mode: IsaacLab env.step/reset do
            # in-place buffer updates that are only legal inside inference_mode,
            # matching rsl_rl + run_isaaclab_eval; SAC update below is normal mode) ---
            with torch.inference_mode():
                norm_obs = self.obs_normalizer(obs, update=False)
                actions = self.actor.explore(norm_obs, dones=dones, deterministic=False)
                next_obs, rewards, dones, infos = env.step(actions.float())
                truncations = infos["time_outs"]
                next_critic_obs = infos["observations"]["critic"]
                # IsaacLab does not expose terminal obs -> next_obs as bootstrap target.
                self.rb.extend(
                    obs, actions, rewards, dones.long(), truncations.long(),
                    next_obs, critic_obs, next_critic_obs,
                )
            obs, critic_obs = next_obs, next_critic_obs

            # --- UTD updates ---
            train_metrics = {}
            if it > cfg.learning_starts:
                bs = max(cfg.batch_size // self.num_envs, 1)
                batches = self._sample_and_prepare_batches(bs, cfg.num_updates)
                for i, data in enumerate(batches):
                    m = self._update_main(data)
                    if cfg.num_updates > 1:
                        if i % cfg.policy_frequency == 1:
                            m.update(self._update_pol(data))
                    elif it % cfg.policy_frequency == 0:
                        m.update(self._update_pol(data))
                    with torch.no_grad():
                        src = [p.data for p in self.qnet.parameters()]
                        tgt = [p.data for p in self.qnet_target.parameters()]
                        torch._foreach_mul_(tgt, 1.0 - cfg.tau)
                        torch._foreach_add_(tgt, src, alpha=cfg.tau)
                    train_metrics = m

            # --- periodic train log ---
            if it % cfg.log_interval == 0 and train_metrics:
                log_d = {
                    "train/qf_loss": float(train_metrics.get("qf_loss", 0.0)),
                    "train/qf_max": float(train_metrics.get("qf_max", 0.0)),
                    "train/qf_min": float(train_metrics.get("qf_min", 0.0)),
                    "train/actor_loss": float(train_metrics.get("actor_loss", 0.0)),
                    "train/alpha": float(self.log_alpha.exp().item()),
                    "train/policy_entropy": float(train_metrics.get("policy_entropy", 0.0)),
                    "train/action_std": float(train_metrics.get("action_std", 0.0)),
                    "train/truncation_frac": float(truncations.float().mean().item()),
                    "train/buffer_reward": float(train_metrics.get("buffer_rewards", 0.0)),
                }
                if wandb.run is not None:
                    wandb.log(log_d, step=it)
                if it % (cfg.log_interval * 20) == 0:
                    print(
                        f"[sac-train] it={it} qf_max={log_d['train/qf_max']:.3f} "
                        f"qf_min={log_d['train/qf_min']:.3f} alpha={log_d['train/alpha']:.5f} "
                        f"ent={log_d['train/policy_entropy']:.3f} astd={log_d['train/action_std']:.3f} "
                        f"trunc={log_d['train/truncation_frac']:.3f}",
                        flush=True,
                    )

            # --- env_steps + eval cadence ---
            env_steps = (it + 1) * self.num_envs
            if env_steps >= next_eval:
                self._run_eval(env_steps, eval_step=it + 1)
                next_eval = env_steps + (EVAL_INTERVAL_EARLY if env_steps < EVAL_PHASE_SWITCH else EVAL_INTERVAL_LATE)
                with torch.inference_mode():
                    obs, extras = env.reset()
                critic_obs = extras["observations"]["critic"]
                dones = torch.zeros(self.num_envs, dtype=torch.long, device=device)
