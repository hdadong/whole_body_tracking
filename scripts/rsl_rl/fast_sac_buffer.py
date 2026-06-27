"""FastSAC replay buffer + obs normalizer (ported from holosoma fast_sac_utils.py).

Removed holosoma-only deps: `tensordict.TensorDict` (extend takes explicit args,
sample returns a plain nested dict) and `torch.distributed` (single-process only).
The n-step return logic and asymmetric actor/critic storage are kept intact.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """Circular replay buffer with n-step returns and asymmetric observations."""
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.actions = torch.zeros((n_env, buffer_size, n_act), device=device, dtype=torch.float)
        self.rewards = torch.zeros((n_env, buffer_size), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.next_observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.critic_observations = torch.zeros((n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float)
        self.next_critic_observations = torch.zeros(
            (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
        )
        self.ptr = 0

    def extend(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        next_observations: torch.Tensor,
        critic_observations: torch.Tensor,
        next_critic_observations: torch.Tensor,
    ):
        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        self.critic_observations[:, ptr] = critic_observations
        self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Sample n_env * batch_size transitions. Returns a plain nested dict."""
        if self.n_steps == 1:
            indices = torch.randint(0, min(self.buffer_size, self.ptr), (self.n_env, batch_size), device=self.device)
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
            next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
            rewards = torch.gather(self.rewards, 1, indices).reshape(self.n_env * batch_size)
            dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
            truncations = torch.gather(self.truncations, 1, indices).reshape(self.n_env * batch_size)
            effective_n_steps = torch.ones_like(dones)
            critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
            critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                self.n_env * batch_size, self.n_critic_obs
            )
            next_critic_observations = torch.gather(self.next_critic_observations, 1, critic_obs_indices).reshape(
                self.n_env * batch_size, self.n_critic_obs
            )
        else:
            if self.ptr >= self.buffer_size:
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1])
                indices = torch.randint(0, self.buffer_size, (self.n_env, batch_size), device=self.device)
            else:
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(0, max_start_idx, (self.n_env, batch_size), device=self.device)
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
            critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
            critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                self.n_env * batch_size, self.n_critic_obs
            )
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size
            all_rewards = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_dones = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones_shifted = torch.cat([torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2)
            done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=2)
            effective_n_steps = done_masks.sum(2)
            discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
            masked_rewards = all_rewards * done_masks
            discounted_rewards = masked_rewards * discounts.view(1, 1, -1)
            n_step_rewards = discounted_rewards.sum(dim=2)
            first_done = torch.argmax((all_dones > 0).float(), dim=2)
            first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)
            final_indices = torch.minimum(first_done, first_trunc)
            final_next_obs_indices = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(-1)
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)
            final_next_critic_observations = self.next_critic_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
            )
            next_critic_observations = final_next_critic_observations.reshape(
                self.n_env * batch_size, self.n_critic_obs
            )
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(self.n_env * batch_size, self.n_obs)
            if self.ptr >= self.buffer_size:
                self.truncations[:, current_pos - 1] = curr_truncations

        return {
            "observations": observations,
            "actions": actions,
            "critic_observations": critic_observations,
            "next": {
                "rewards": rewards,
                "dones": dones,
                "truncations": truncations,
                "observations": next_observations,
                "effective_n_steps": effective_n_steps,
                "critic_observations": next_critic_observations,
            },
        }


class EmpiricalNormalization(nn.Module):
    """Normalize mean/variance of values based on empirical statistics (single-process)."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return
        global_batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + global_batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
