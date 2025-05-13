import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class DoubleCritic(torch.nn.Module):
    max_length_tokenizer: int = 512
    def __init__(
        self,
        device: torch.device,
        critic_lm: str,
        cache_dir: str,
        in_dim: int,
        out_dim: int,
        use_lora: bool = False,
        use_bfloat16: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_double_value_network: bool = True,
    ):
        super(DoubleCritic, self).__init__()
        self.device = device
        self.use_double_value_network = use_double_value_network
        if use_bfloat16:
            # we can accommodate for additional tokens, but if the model is trained for
            # smaller token lengths, the final layer (summarizes the sentence) may require more
            # parameters, ignore_mismatched_sizes adds randomly initialized parameters to the model
            # for this purpose.
            self.base_lm = AutoModel.from_pretrained(
                critic_lm,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            self.base_lm = AutoModel.from_pretrained(
                critic_lm,
                cache_dir=cache_dir,
            ).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            # using encoder only model to extract features
            lora_config = LoraConfig(
                r=lora_rank,
                target_modules="all-linear",
                task_type=TaskType.FEATURE_EXTRACTION,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.base_lm = get_peft_model(self.base_lm, lora_config)
            print("Using LoRA for the critic")
            self.base_lm.print_trainable_parameters()
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            critic_lm, cache_dir=cache_dir
        )
        self.base_tokenizer.truncation_side = "left"
        model_kwargs = {
            'device': device,
        }
        if use_bfloat16:
            model_kwargs['dtype'] = torch.bfloat16
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        ).to(**model_kwargs)
        self.critic2 = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        ).to(**model_kwargs)
        self.v_critic1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        ).to(**model_kwargs)
        # most RL methods do not train two value networks.
        # Adding this as a flag here as it used in the original implementation
        if self.use_double_value_network:
            self.v_critic2 = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
            ).to(**model_kwargs)

    def forward(self, observation, action, detach_model=False):
        # Tokenize observations and convert to ids (int)
        obs_ids = self.base_tokenizer(
            observation,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length_tokenizer,
            truncation=True,
        ).to(self.device)
        # extract the pooler_output --> last layer output of the first input in the sequence (classification token)
        # should capture the semantic meaning of the sentence/input.
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
        # Tokenize actions and convert to ids (int)
        action_ids = self.base_tokenizer(
            action,
            padding=True,  # pads to the longest sequence in the batch
            return_tensors="pt",  # returns a torch tensor
            truncation=True,  # truncates by the maximum length specified by the user
            max_length=self.max_length_tokenizer,  #  length of truncation
        ).to(self.device)
        # breakpoint()
        if detach_model:
            with torch.no_grad():
                action_states = self.base_lm(**action_ids).pooler_output
        else:
            action_states = self.base_lm(**action_ids).pooler_output
        # merge to define state action pairs for the critics
        q_states = torch.cat([lm_states, action_states], dim=1)
        if self.use_double_value_network:
            return (
                self.critic1(q_states),
                self.critic2(q_states),
                self.v_critic1(lm_states),
                self.v_critic2(lm_states),
            )
        else:
            # if using only one critic, return its value twice
            v = self.v_critic1(lm_states)
            return (
                self.critic1(q_states),
                self.critic2(q_states),
                v,
                v,
            )
