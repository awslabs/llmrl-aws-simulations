import copy

import torch
from transformers import PreTrainedTokenizer
from accelerate import Accelerator
from agents.bc_agent import BCAgent
from torch.utils.data import DataLoader
from data_utils import DummyDataset, ReplayBuffer
from trainers.dummy_trainer import DummyTrainer


def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class BCTrainer(DummyTrainer):
    def __init__(
        self,
        agent: BCAgent,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
        lm_lr: float = 1e-5,
        epochs: int = 3,
        max_grad_norm: float = 0.01,
        grad_accum_steps: int = 8,
        kl_weight: float = 0.1,
        kl_beta: float = 1.0,
        use_distribution_averaging: bool = False,
    ):
        """
        beta: coefficient for the bc loss
        """
        super().__init__(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
        self.lm_optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=lm_lr)
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.step = 0
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.prepare()

    def prepare(self):
        self.agent.prepare()
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)

    def actor_loss(self, observation, action, **kwargs):
        info = {}
        log_probs = self.agent.get_log_prob(observation=observation, action=action)
        loss = -log_probs.mean()
        info["bc.loss"] = loss.detach().cpu().item()
        if self.regularize_with_anchor:
            # we want to approximate grad_theta E_{a ~ Q}[f(Q(a), P(a))], e.g., f = -log(P(a)/Q(a))
            # where Q = Q(|\theta)
            # = grad_Q \sum_a f(Q(a), P(a)) Q(a) = \sum_a (grad_theta f(Q(a), P(a))) Q(a) + f(Q(a), P(a)) grad_theta Q
            # = \sum_a (grad_theta f(Q(a), P(a))) + f(Q(a), P(a)) (grad_theta log Q) Q
            # = E_{a ~ Q}[grad_theta f(Q(a), P(a))) + f(Q(a), P(a)) grad_theta log Q]
            # L(\theta) ~ f(Q(a), P(a))) + f(Q(a), P(a))).detach() log Q
            with torch.no_grad():
                pi_action = self.agent.get_action(observation=copy.deepcopy(observation))
            distance = self.anchor_distance(
                observation=observation, pi_action=pi_action
            )
            info.update({"distance": distance.mean().detach().cpu().item()})
            log_pi = self.agent.get_log_prob(observation=observation, action=pi_action)
            distance_loss = (distance.detach() * log_pi + distance).mean()
            loss = loss + distance_loss
        self.accelerator.backward(loss)
        return info

    def update(self, replay_buffer: ReplayBuffer, no_update_actor: bool = False):
        self.step += 1
        info = {}
        info_list = []
        # update actor
        if not no_update_actor:
            action_bsize = (
                1
                if "llama" in self.accelerator.unwrap_model(self.agent).policy_lm
                else replay_buffer.batch_size
            )
            for _ in range(self.epochs):
                self.lm_optimizer.zero_grad()
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
                for batch in dataloader:
                    info_list.append(self.actor_loss(**batch))
                self.accelerator.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        return info

    def save(self, path: str):
        model_dict = self.agent.get_model_state_dict()
        model_dict.update({"lm_optimizer_state_dict": self.lm_optimizer.state_dict()})
        torch.save(model_dict, path)

    def load(self, path: str, map_location=None):
        if map_location is not None:
            checkpoint = torch.load(path, map_location=map_location)
        else:
            checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.lm_optimizer.load_state_dict(checkpoint["lm_optimizer_state_dict"])
        return self.agent
