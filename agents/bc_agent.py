import torch
from accelerate import Accelerator
from typing import Optional, Union
from agents.dummy_agent import DummyAgent
from agents.linear_scheduler import LinearScheduler


class BCAgent(DummyAgent):
    def __init__(
        self,
        device: torch.device,
        accelerator: Accelerator,
        policy_lm: str = "gpt2",
        cache_dir: str = "~/.cache",
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
        attn_implementation: str = "sdpa",
        beam_size: int = 1,
    ):
        super(BCAgent, self).__init__(
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

    def get_action(self, observation, **kwargs):
        return self._get_action(observation)
