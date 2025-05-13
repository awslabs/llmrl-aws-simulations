import torch
from agents.critic import DoubleCritic
from accelerate import Accelerator
from typing import Optional, Union
from agents.linear_scheduler import LinearScheduler
from agents.dummy_agent import DummyAgent


class CHAIAgent(DummyAgent):
    def __init__(
        self,
        device: torch.device,
        accelerator: Accelerator,
        policy_lm: str = "gpt2",
        critic_lm: str = "roberta-base",
        cache_dir: str = "~/.cache",
        template: Optional[str] = None,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: Union[float, LinearScheduler] = 1.0,
        eval_temperature: Union[float, LinearScheduler] = 0.1,
        eos_str: str = "\n",
        in_dim: int = 768,
        out_dim: int = 1,
        num_action_selection_samples: int = 5,
        max_length_tokenizer: int = 512,
        use_bfloat16: bool = False,
        use_lora: bool = False,
        use_lora_for_critic: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_double_value_network: bool = False,
        quantization: Optional[int] = None,
        use_better_transformer: bool = False,
        sampling_strategy: str = "greedy",
        attn_implementation: str = "sdpa",
        beam_size: int = 1,
    ):
        assert do_sample, "do sample must be True for CHAI agent"
        super(CHAIAgent, self).__init__(
            device=device,
            accelerator=accelerator,
            policy_lm=policy_lm,
            cache_dir=cache_dir,
            template=template,
            use_lora=use_lora,
            do_sample=do_sample,
            temperature=temperature,
            eval_temperature=eval_temperature,
            max_new_tokens=max_new_tokens,
            max_length_tokenizer=max_length_tokenizer,
            use_bfloat16=use_bfloat16,
            eos_str=eos_str,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantization=quantization,
            use_better_transformer=use_better_transformer,
            attn_implementation=attn_implementation,
            beam_size=beam_size,
        )
        assert sampling_strategy in ["greedy", "random"]
        self.sampling_strategy = sampling_strategy
        self.use_double_value_network = use_double_value_network

        self.critic = DoubleCritic(
            device,
            critic_lm=critic_lm,
            cache_dir=cache_dir,
            in_dim=in_dim,
            out_dim=out_dim,
            use_bfloat16=use_bfloat16,
            use_lora=use_lora_for_critic,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_double_value_network=use_double_value_network,
        )
        self.target_critic = DoubleCritic(
            device,
            critic_lm=critic_lm,
            cache_dir=cache_dir,
            in_dim=in_dim,
            out_dim=out_dim,
            use_bfloat16=use_bfloat16,
            use_lora=use_lora_for_critic,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_double_value_network=use_double_value_network,
        )
        # set target critic equal to critic at initialization
        self.soft_update_target_critic(1)
        self.num_action_selection_samples = num_action_selection_samples

    def prepare(self):
        # prepares the policy
        super().prepare()
        # preparing the critic
        self.critic, self.target_critic = self.accelerator.prepare(
            self.critic, self.target_critic
        )

    def get_action(self, observation, return_q_value: bool = False):
        # samples actions and returns the one that have the highest Q value
        # batch_actions = []
        # batch_qs = []
        # sample num_action_selection_samples different actions and store their q values
        # multiply the observations to do batched inference with the policy
        if self.num_action_selection_samples > 1:
            old_sample = self.do_sample
            self.do_sample = True
            multi_obs = [[ob] * self.num_action_selection_samples for ob in observation]
            multi_obs = sum(multi_obs, [])
            actions = self._get_action(multi_obs)
            q1, q2, _, _ = self.target_critic(multi_obs, actions, detach_model=True)
            q = torch.min(q1, q2)
            # split the q values and actions for each query.
            batch_qs = q.reshape(-1, self.num_action_selection_samples)
            batch_actions = [
                actions[
                    i
                    * self.num_action_selection_samples : (i + 1)
                    * self.num_action_selection_samples
                ]
                for i in range(len(observation))
            ]

            # for _ in range(self.num_action_selection_samples):
            #    actions = self._get_action(observation)
            #    q1, q2, _, _ = self.target_critic(observation, actions, detach_model=True)
            #    qs = torch.minimum(q1, q2)
            #    batch_actions.append(actions)
            #    batch_qs.append(qs.reshape(1, -1))
            # pick the actions with the largest q values across different batches
            # batch_qs = torch.cat(batch_qs, dim=0)
            # select indices based on Q values and return actions
            if self.sampling_strategy == "greedy":
                # selected_ids = torch.argmax(batch_qs, dim=0).cpu().numpy()
                selected_ids = torch.argmax(batch_qs, dim=-1).cpu().numpy()
            else:
                raise NotImplementedError
            self.do_sample = old_sample
            selected_actions = []
            qs = []
            for i, idx in enumerate(selected_ids):
                # selected_actions.append(batch_actions[idx][i])
                # qs.append(batch_qs[idx][i])
                selected_actions.append(batch_actions[i][idx])
                qs.append(batch_qs[i][idx])
            if return_q_value:
                return selected_actions, torch.stack(qs)
            else:
                return selected_actions
        else:
            assert (
                not return_q_value
            ), "Q values are not calculate when only sample is generated"
            return self._get_action(observation)

    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_parameters(self):
        model_state_dict = super().get_model_state_dict()
        model_state_dict.update(
            {
                "critic_state_dict": self.accelerator.unwrap_model(
                    self.critic
                ).state_dict(),
                "target_critic_state_dict": self.accelerator.unwrap_model(
                    self.target_critic
                ).state_dict(),
            }
        )
        return model_state_dict

