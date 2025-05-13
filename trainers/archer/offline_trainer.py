import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Union, Optional
from data_utils.utils import DummyDataset, ReplayBuffer
from agents.archer_agent import ArcherAgent
import copy
from trainers.dummy_trainer import DummyTrainer


def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class OfflineArcherTrainer(DummyTrainer):
    def __init__(
        self,
        agent: ArcherAgent,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizer,
        critic_lr: float = 1e-3,
        lm_lr: float = 1e-5,
        grad_accum_steps: int = 8,
        gamma: float = 0.9,
        tau: float = 0.1,
        epochs: int = 3,
        max_grad_norm: float = 0.01,
        actor_epochs: int = 3,
        kl_weight: float = 0.1,
        kl_beta: float = 1.0,
        use_distribution_averaging: bool = False,
        iql_tau: float = 0.9,
        advantage_beta: float = 2.0,
        weight_max: float = 20.0,
    ):
        """
        beta: coefficient for the bc loss
        """
        assert iql_tau < 1 and iql_tau >= 0.5, "iql_tau must be in (0.5, 1)"
        super().__init__(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
        self.use_double_value_network = self.agent.use_double_value_network
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.critic_optimizer = torch.optim.Adam(
            agent.critic.parameters(), lr=critic_lr
        )
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.iql_tau = iql_tau
        self.advantage_beta = advantage_beta
        self.max_grad_norm = max_grad_norm
        self.log_weight_max = np.log(weight_max)
        self.agent.prepare()
        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(
            self.critic_optimizer, self.lm_optimizer
        )

    def v_loss(self, target: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        u = target - value
        # Loss = |\tau - (u < 0)| * u^2, tau \in [0.5, 1)
        # if u >= 0 -> L = \tau u^2, if u < 0 -> L = (1 - \tau) u^2
        # see line 351 - 3543
        # here: https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/main/src/models/iql_model.py
        loss = (u >= 0).int() * self.iql_tau * torch.square(u) + (u < 0).int() * (
            1 - self.iql_tau
        ) * torch.square(u)
        loss = loss.mean()
        return loss

    def critic_loss(
        self,
        observation: List[str],
        action: List[str],
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
        mc_return: np.ndarray,
        **kwargs
    ):
        info = {}
        reward = (
            torch.Tensor(reward)
            .to(
                self.accelerator.unwrap_model(self.agent.model).device,
                dtype=self.accelerator.unwrap_model(self.agent.model).dtype,
            )
            .flatten()
        )
        done = (
            torch.Tensor(done)
            .to(
                self.accelerator.unwrap_model(self.agent.model).device,
                dtype=self.accelerator.unwrap_model(self.agent.model).dtype,
            )
            .flatten()
        )
        # get targets for Q learning
        with torch.no_grad():
            # get Q value for the policy
            pi_action = self.agent.get_action(copy.deepcopy(observation))
            if self.regularize_with_anchor:
                # get distance penalty to anchor
                distance = self.anchor_distance(
                    observation=observation, pi_action=pi_action
                )
                reward = reward - distance
                info["distance"] = distance.mean().detach().cpu().item()
            # target_q1, target_q2 = self.agent.get_q(observation, pi_action, detach_model=False)
            target_q1, target_q2, _, _ = self.agent.target_critic(
                copy.deepcopy(observation), pi_action, detach_model=False
            )
            target_q = torch.minimum(target_q1, target_q2)
            # action is ignored when computing v1 and v2
            _, _, target_v1, target_v2 = self.agent.target_critic(
                next_observation, copy.deepcopy(action)
            )
            target_v1 = reward + (1 - done) * target_v1.flatten() * self.gamma
            target_v2 = reward + (1 - done) * target_v2.flatten() * self.gamma

        # Get Q values for TD error
        q1, q2, v1, v2 = self.agent.critic(observation, action, detach_model=False)
        # print("finish one forward pass")
        q1 = q1.flatten()
        q2 = q2.flatten()
        v1 = v1.flatten()
        target_q = target_q.flatten()
        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v2)
        # use the IQL loss for the value function to compensate for overestimation.
        # effectively, weights data points where we have a higher V more.
        v1_loss = self.v_loss(target=target_q, value=v1)
        if self.use_double_value_network:
            v2 = v2.flatten()
            v2_loss = self.v_loss(target=target_q, value=v2)
            self.accelerator.backward((q1_loss + q2_loss + v1_loss + v2_loss))
            q1_loss, q2_loss, v1_loss, v2_loss = (
                q1_loss.detach().cpu(),
                q2_loss.detach().cpu(),
                v1_loss.detach().cpu(),
                v2_loss.detach().cpu(),
            )
            q1, q2, v1, v2, target_q1, target_q2 = (
                q1.detach().cpu(),
                q2.detach().cpu(),
                v1.detach().cpu(),
                v2.detach().cpu(),
                target_q1.detach().cpu(),
                target_q2.detach().cpu(),
            )
        else:
            self.accelerator.backward((q1_loss + q2_loss + v1_loss))
            q1_loss, q2_loss, v1_loss, v2_loss = (
                q1_loss.detach().cpu(),
                q2_loss.detach().cpu(),
                v1_loss.detach().cpu(),
                v1_loss.detach().cpu(),
            )
            q1, q2, v1, v2, target_q1, target_q2 = (
                q1.detach().cpu(),
                q2.detach().cpu(),
                v1.detach().cpu(),
                v1.detach().cpu(),
                target_q1.detach().cpu(),
                target_q2.detach().cpu(),
            )

        info.update(
            {
                "q1.loss": q1_loss,
                "q2.loss": q2_loss,
                "v1.loss": v1_loss,
                "v2.loss": v2_loss,
                "q1.mean": torch.mean(q1),
                "q1.min": torch.min(q1),
                "q1.max": torch.max(q1),
                "q1.std": torch.std(q1),
                "q2.mean": torch.mean(q2),
                "q2.max": torch.max(q2),
                "q2.min": torch.min(q2),
                "q2.std": torch.std(q2),
                "v1.mean": torch.mean(v1),
                "v1.min": torch.min(v1),
                "v1.max": torch.max(v1),
                "v1.std": torch.std(v1),
                "v2.mean": torch.mean(v2),
                "v2.max": torch.max(v2),
                "v2.min": torch.min(v2),
                "v2.std": torch.std(v2),
                "target_q1.mean": torch.mean(target_q1),
                "target_q1.min": torch.min(target_q1),
                "target_q1.max": torch.max(target_q1),
                "target_q1.std": torch.std(target_q1),
                "target_q2.mean": torch.mean(target_q2),
                "target_q2.max": torch.max(target_q2),
                "target_q2.min": torch.min(target_q2),
                "target_q2.std": torch.std(target_q2),
            }
        )
        return info

    def actor_loss(
        self,
        observation: List[str],
        action: List[str],
        advantage: Union[np.ndarray, torch.Tensor],
        **kwargs
    ):
        """Get observation and action from the data buffer along with the advantage for this action."""
        # calculate log_prob of the agent's actions
        log_prob = self.agent.get_log_prob(observation, action)
        # calculate score for the action -> if the action has high advantage,
        # add more weight to it in the loss.
        advantage = torch.Tensor(advantage).to(
            self.accelerator.unwrap_model(self.agent.model).device,
            dtype=self.accelerator.unwrap_model(self.agent.model).dtype,
        )
        # makes sure that the score is at most weight_max to avoid instabilities
        coefficient = torch.clamp(
            self.advantage_beta * advantage, max=self.log_weight_max
        )
        score = torch.exp(coefficient)
        loss = -torch.mean(log_prob.flatten() * score)
        advantage = advantage.flatten()
        self.accelerator.backward(loss)
        advantage = advantage.detach().cpu()
        return {
            "policy.loss": loss.detach().cpu().item(),
            "log_prob.mean": log_prob.mean().item(),
            "advantages.mean": advantage.mean(),
            "advantages.max": torch.max(advantage),
            "advantages.min": torch.min(advantage),
            "advantages.std": torch.std(advantage),
        }

    def update(self, replay_buffer: ReplayBuffer, no_update_actor: bool = False):
        self.step += 1
        info = {}
        info_list = []
        with torch.autograd.set_detect_anomaly(True):
            for _ in range(self.epochs):
                data = [
                    replay_buffer.sample(1)
                    for _ in range(self.grad_accum_steps * replay_buffer.batch_size)
                ]
                for d in data:
                    for k, v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(
                    DummyDataset(data), batch_size=replay_buffer.batch_size
                )
                dataloader = self.accelerator.prepare(dataloader)
                self.critic_optimizer.zero_grad()
                # train critic
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(**batch))
                self.accelerator.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                self.agent.soft_update_target_critic(tau=self.tau)
        info.update(dict_mean(info_list))
        info_list = []
        # update actor
        if not no_update_actor:
            print(">>>updating actor")
            # batchsize for the actor set to 1 for mistral due to memory concern
            action_bsize = (
                2 if "mistral" in self.agent.policy_lm else replay_buffer.batch_size
            )
            for _ in range(self.actor_epochs):
                data = [
                    replay_buffer.sample(1)
                    for _ in range(self.grad_accum_steps * replay_buffer.batch_size)
                ]
                for d in data:
                    for k, v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(
                    DummyDataset(data), batch_size=action_bsize, shuffle=False
                )
                dataloader = self.accelerator.prepare(dataloader)
                self.lm_optimizer.zero_grad()
                for batch in dataloader:
                    with torch.no_grad():
                        # get critic value from the data
                        q1, q2, v1, v2 = self.agent.critic(
                            batch["observation"], batch["action"]
                        )
                        q = torch.minimum(q1, q2)
                        # v1, v2 = self.agent.critic(batch["observation"])
                        v = torch.minimum(v1, v2)
                        # calculate utterance level advantage
                        advantages = q - v
                    info_list.append(self.actor_loss(**batch, advantage=advantages))
                self.accelerator.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        return info

    def save(self, path: str):
        model_dict = self.agent.get_model_state_dict()
        model_dict.update(
            {
                "lm_optimizer_state_dict": self.lm_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            }
        )
        torch.save(model_dict, path)

    def load(self, path: str, map_location: Optional[str] = None):
        if map_location is not None:
            checkpoint = torch.load(path, map_location=map_location)
        else:
            checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.agent.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.lm_optimizer.load_state_dict(checkpoint["lm_optimizer_state_dict"])
        return self.agent
