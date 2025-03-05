from __future__ import annotations

import time

import torch
import copy
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from typing import Dict, Union

from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env import VecEnv
from rsl_rl.modules.network import Network
from rsl_rl.storage.storage import Dataset
from rsl_rl.modules.actor_critic import ActorCritic


class DDPG(AbstractDPG):
    """Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the DDPG algorithm by Lillicrap et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    # torch.autograd.set_detect_anomaly(True)
    def __init__(
            self,
            env: VecEnv,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            init_noise_std: float = 1.,
            noise_decay_rate: float = 0.998,
            **kwargs,
    ) -> None:

        super().__init__(env, **kwargs)

        self.actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)

        self.target_actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.target_critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Exploration noise
        self.noise_decay_rate = noise_decay_rate
        self.init_noise_std = torch.full((self._action_size,), init_noise_std, device=self.device)
        self.std = nn.Parameter(init_noise_std * torch.ones(self._action_size, device=self.device))
        self.min_noise_std = torch.zeros(self._action_size, device=self.device)
        self.time_cnt = 0

        self._register_serializable(
            "actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"
        )

        self.to(self.device)

    def eval_mode(self) -> DDPG:
        super().eval_mode()

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        return self

    def act(self, obs, critic_obs):
        return self.nn_act(obs).detach()

    def to(self, device: str) -> DDPG:
        """Transfers agent parameters to device."""
        super().to(device)

        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

        return self

    def train_mode(self) -> DDPG:
        super().train_mode()

        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        return self

    def update(self, dataset: Dataset) -> [float, float]:
        super().update(dataset)

        if not self.initialized:
            return {}

        total_actor_loss = torch.zeros(self._batch_count)
        total_critic_loss = torch.zeros(self._batch_count)

        for idx, batch in enumerate(self.storage.batch_generator(self._batch_size, self._batch_count)):
            actor_obs = batch["actor_observations"]
            critic_obs = batch["critic_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"].int()

            target_actor_prediction = self._process_actions(self.target_actor.forward(actor_next_obs))
            target_critic_prediction = self.target_critic.forward(
                self._critic_input(critic_next_obs, target_actor_prediction)
            )
            target = rewards + self._discount_factor * (1 - dones) * target_critic_prediction
            prediction = self.critic.forward(self._critic_input(critic_obs, actions))

            # Critic Loss
            critic_loss = (prediction - target).pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            evaluation = self.critic.forward(
                self._critic_input(critic_obs, self._process_actions(self.actor.forward(actor_obs)))
            )

            # Actor Loss
            actor_loss = -evaluation.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Soft update Target ActorCritic
            self._update_target(self.actor, self.target_actor)
            self._update_target(self.critic, self.target_critic)

            total_actor_loss[idx] = actor_loss.item()
            total_critic_loss[idx] = critic_loss.item()

        stats = {"actor": total_actor_loss.mean().item(), "critic": total_critic_loss.mean().item()}
        return stats

    def to_transition(self, prev_obs, obs, actions, rewards, dones, infos):
        return {
            'actor_observations': prev_obs.clone(),
            'critic_observations': prev_obs.clone(),
            'actions': actions.clone(),
            'rewards': rewards.clone(),
            'next_actor_observations': obs.clone(),
            'next_critic_observations': obs.clone(),
            'dones': dones.clone(),
            'timeouts': infos['time_outs'].clone()
        }

    def update_distribution(self, observations):
        mean = self.actor(observations)

        # Annealing for noise
        # with torch.no_grad():
        #     if self.time_cnt % 100:
        #         self.std = torch.max(self.min_noise_std, self.std * self.noise_decay_rate)
        #     self.time_cnt += 1

        self.distribution = Normal(mean, torch.clamp(self.std, min=1e-6))

    def nn_act(self, observations, **kwargs):
        self.update_distribution(observations)

        return torch.clamp(self.distribution.sample(), min=0., max=1.)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        value = torch.unsqueeze(value, 1)
        return value

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def reset(self, dones=None):
        pass
