import torch
from agents.critic import DoubleCritic
from accelerate import Accelerator
from typing import Optional, Union
from agents.dummy_agent import DummyAgent
from agents.linear_scheduler import LinearScheduler


class ArcherAgent(DummyAgent):
    def __init__(
        self,
        device: torch.device,
        accelerator: Accelerator,
        policy_lm: str = "gpt2",
        critic_lm: str = "roberta-base",
        cache_dir: str = "~/.cache",
        dropout: float = 0.5,
        template: Optional[str] = None,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: Union[float, LinearScheduler] = 1.0,
        eval_temperature: Union[float, LinearScheduler] = 0.1,
        eos_str: str = "\n",
        max_length_tokenizer: int = 512,
        use_lora: bool = False,
        use_bfloat16: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        quantization: Optional[int] = None,
        use_better_transformer: bool = False,
        use_lora_for_critic: bool = False,
        in_dim: int = 768,
        use_double_value_network: bool = True,
        attn_implementation: str = "sdpa",
        beam_size: int = 1,
    ):
        super(ArcherAgent, self).__init__(
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
        self.use_double_value_network = use_double_value_network
        self.critic = DoubleCritic(
            device=device,
            critic_lm=critic_lm,
            cache_dir=cache_dir,
            in_dim=in_dim,
            out_dim=1,
            use_bfloat16=use_bfloat16,
            use_lora=use_lora_for_critic,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_double_value_network=use_double_value_network,
        )
        self.target_critic = DoubleCritic(
            device=device,
            critic_lm=critic_lm,
            cache_dir=cache_dir,
            in_dim=in_dim,
            out_dim=1,
            use_bfloat16=use_bfloat16,
            use_lora=use_lora_for_critic,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_double_value_network=use_double_value_network,
        )
        self.soft_update_target_critic(1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def prepare(self):
        # prepares the policy
        super().prepare()
        # preparing the critic
        self.critic, self.target_critic = self.accelerator.prepare(
            self.critic, self.target_critic
        )

    def get_action(self, observation, **kwargs):
        return self._get_action(observation)

    # def get_log_prob(self, observation: List[str], action: List[str]):
    #     if self.template is not None:
    #         observation = [self.template.replace("{obs}", obs) for obs in observation]
    #     obs_ids = self.tokenizer(
    #         observation,
    #         return_tensors="pt",
    #         padding=True,
    #         max_length=self.max_length_tokenizer,
    #         truncation=True,
    #     ).to(self.device)
    #     action_ids = self.tokenizer(
    #         action,
    #         return_tensors="pt",
    #         padding=True,
    #         max_length=self.max_length_tokenizer,
    #         truncation=True,
    #     ).to(self.device)
    #     # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
    #     # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
    #     input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
    #     # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
    #     attention_mask = torch.cat(
    #         [obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1
    #     )
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     values = None
    #     if isinstance(outputs, Tuple):
    #         values, outputs = outputs
    #     prediction_probs = self.softmax(outputs.logits)
    #     selected_prediction_probs = torch.take_along_dim(
    #         prediction_probs[:, obs_ids["attention_mask"].size(1) - 1 : -1],
    #         action_ids["input_ids"].unsqueeze(2),
    #         dim=2,
    #     ).squeeze(2)
    #     if values is not None:
    #         return (
    #             values[:, obs_ids["attention_mask"].size(1) - 1 : -1],
    #             torch.log(selected_prediction_probs) * action_ids["attention_mask"],
    #             action_ids["attention_mask"],
    #         )
    #     else:
    #         return torch.sum(
    #             torch.log(selected_prediction_probs) * action_ids["attention_mask"],
    #             dim=1,
    #         )

    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_state_dict(self):
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
