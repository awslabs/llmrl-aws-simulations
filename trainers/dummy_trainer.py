from accelerate import Accelerator
from transformers import PreTrainedTokenizer
import copy
import torch
from agents.dummy_agent import DummyAgent


class DummyTrainer(object):
    def __init__(
        self,
        agent: DummyAgent,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizer,
        kl_weight: float = 0.1,
        kl_beta: float = 1.0,
        use_distribution_averaging: bool = False,
        *args,
        **kwargs,
    ):
        assert kl_weight >= 0 and 0 <= kl_beta <= 1
        self.agent = agent
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.kl_weight = kl_weight
        self.kl_beta = kl_beta
        self.use_distribution_averaging = use_distribution_averaging
        self.regularize_with_anchor = kl_weight > 0.0
        if self.regularize_with_anchor:
            assert (
                self.agent.has_anchor
            ), "Agent must have an anchor to regularize with initial policy"


    def anchor_distance(self, observation, pi_action):
        assert (
            self.regularize_with_anchor
        ), "KL weight is 0.0, no regularization is required."
        # in the following, Q is the agent distribution and P the target distribution
        if self.kl_beta == 1.0:
            # KL(Q||P), pi_action ~ Q
            log_agent = self.agent.get_log_prob(observation, pi_action)  # log(Q)
            with torch.no_grad():
                log_anchor = self.agent.get_log_prob_for_model(
                    observation, pi_action, model_type="anchor"
                )  # log(P)
            bw_kl = log_agent - log_anchor
            return bw_kl * self.kl_weight
        elif self.kl_beta == 0.0:
            # KL(P||Q)
            with torch.no_grad():
                target_action = self.agent.get_action_for_model(
                    copy.deepcopy(observation), model_type="anchor"
                )
                log_anchor = self.agent.get_log_prob_for_model(
                    observation, target_action, model_type="anchor"
                )  # log(P)
            # log(P) - log(Q)
            log_agent = self.agent.get_log_prob(observation, target_action)  # log(Q)
            fw_kl = log_anchor - log_agent
            return fw_kl * self.kl_weight
        else:
            if self.use_distribution_averaging:
                # we calculate: beta * KL(P|| beta * P + (1-beta) * Q) + (1-beta) * KL(Q || beta * P + (1-beta) * Q)
                # calculate KL(Q || beta * P + (1-beta) * Q) (backward)
                # pi_action ~ Q
                log_agent = self.agent.get_log_prob(observation, pi_action)  # log(Q)
                with torch.no_grad():
                    log_anchor = self.agent.get_log_prob_for_model(
                        observation, pi_action, model_type="anchor"
                    )  # log(P)
                P, Q = torch.exp(log_anchor), torch.exp(log_agent)
                # log(T) = log(beta * P + (1 - beta) * Q) = log(beta(P-Q) + Q)
                log_target = torch.log(self.kl_beta * (P - Q) + Q)
                # (log(Q) - log(T)) * (1-beta)
                bw_kl = (log_agent - log_target) * (1 - self.kl_beta)
                with torch.no_grad():
                    target_action = self.agent.get_action_for_model(
                        copy.deepcopy(observation), model_type="anchor"
                    )
                    log_anchor = self.agent.get_log_prob_for_model(
                        observation, target_action, model_type="anchor"
                    )  # log(P)
                log_agent = self.agent.get_log_prob(
                    observation, target_action
                )  # log(Q)
                P, Q = torch.exp(log_anchor), torch.exp(log_agent)
                # log(T) = log(beta * P + (1 - beta) * Q) = log(beta(P-Q) + Q)
                log_target = torch.log(self.kl_beta * (P - Q) + Q)
                # fw_kl = beta * (log(P) - log(T))
                fw_kl = (log_anchor - log_target) * self.kl_beta
                return (bw_kl + fw_kl) * self.kl_weight
            else:
                # pi_action ~ Q
                log_agent = self.agent.get_log_prob(observation, pi_action)  # log(Q)
                with torch.no_grad():
                    log_anchor = self.agent.get_log_prob_for_model(
                        observation, pi_action, model_type="anchor"
                    )  # log(P)
                bw_kl = (log_agent - log_anchor) * self.kl_beta

                with torch.no_grad():
                    target_action = self.agent.get_action_for_model(
                        copy.deepcopy(observation), model_type="anchor"
                    )
                    log_anchor = self.agent.get_log_prob_for_model(
                        observation, target_action, model_type="anchor"
                    )  # log(Q)
                log_agent = self.agent.get_log_prob(
                    observation, target_action
                )  # log(Q)
                fw_kl = (log_anchor - log_agent) * (1 - self.kl_beta)
                return (bw_kl + fw_kl) * self.kl_weight
