import torch
import transformers
from agents import BCAgent, CHAIAgent, ArcherAgent
from agents.linear_scheduler import LinearScheduler
from trainers.training_loop import offpolicy_train_loop
from utils import colorful_print
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
import os
from envs.env_utils import get_envs_for_training
from transformers import set_seed
import torch
import numpy as np


transformers.logging.set_verbosity_error()


def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg="red")
    try:
        from huggingface_hub import login

        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")
    init_process_kwarg = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(kwargs_handlers=[init_process_kwarg, ddp_kwargs])
    device = accelerator.device
    temperature = LinearScheduler(
        start=config.temp_start, end=config.temp_end, total_steps=config.temp_steps
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_seed(config.seed)

    # load environment
    os.makedirs(config.save_path, exist_ok=True)
    if config.env_name == "aws_env":
        # initialize environment in the main process, which will then
        # have access to all servers
        train_env, eval_env = get_envs_for_training(
            num_train_envs=config.num_envs,
            num_eval_envs=config.num_eval_envs,
            seed=config.seed,
            port_number=config.port_number,
            env_kwarg={
                "reset_user_upon_done": True,
                "return_state_as_text": True,
                "max_conversation_length": config.max_conversation_length,
                "log_path": config.save_path,
                "log_every": config.log_every,
                "env_type": config.env_type,
            },
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
        )
    else:
        raise NotImplementedError("Environment not implemented.")
    decode_f = lambda x: x
    # load decision model
    if config.agent_type.lower() == "online_filteredbc":
        print(">>> Using BC agent")
        agent = BCAgent(
            device=device,
            accelerator=accelerator,
            temperature=temperature,
            do_sample=config.do_sample,
            policy_lm=config.policy_lm,
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            max_length_tokenizer=config.max_length_tokenizer,
            use_lora=config.use_lora,
            use_bfloat16=config.use_bfloat16,
            eos_str=config.eos_str,
            use_better_transformer=config.use_better_transformer,
            attn_implementation=config.attn_implementation,
            eval_temperature=config.eval_temperature,
            beam_size=config.beam_size,
            # quantization=config.quantization,
        )

    elif config.agent_type.lower() == "chai":
        print(">>> Using CHAI agent")
        agent = CHAIAgent(
            device=device,
            accelerator=accelerator,
            temperature=temperature,
            do_sample=config.do_sample,
            policy_lm=config.policy_lm,
            critic_lm=config.critic_lm,
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            eos_str=config.eos_str,
            max_length_tokenizer=config.max_length_tokenizer,
            use_lora=config.use_lora,
            use_bfloat16=config.use_bfloat16,
            use_lora_for_critic=config.use_lora_for_critic,
            use_double_value_network=False,
            use_better_transformer=config.use_better_transformer,
            attn_implementation=config.attn_implementation,
            eval_temperature=config.eval_temperature,
            beam_size=config.beam_size,
            # quantization=config.quantization,
        )
        # if use chai, do not update the actor
        config.warmup_iter = config.iterations

    elif config.agent_type.lower() == "archer":
        print(">>> Using ArCHer agent")
        agent = ArcherAgent(
            device=device,
            accelerator=accelerator,
            temperature=temperature,
            do_sample=config.do_sample,
            policy_lm=config.policy_lm,
            critic_lm=config.critic_lm,
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            eos_str=config.eos_str,
            max_length_tokenizer=config.max_length_tokenizer,
            use_lora=config.use_lora,
            use_bfloat16=config.use_bfloat16,
            use_lora_for_critic=config.use_lora_for_critic,
            use_double_value_network=False,
            use_better_transformer=config.use_better_transformer,
            attn_implementation=config.attn_implementation,
            eval_temperature=config.eval_temperature,
            beam_size=config.beam_size,
            # quantization=config.quantization,
        )
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)[
            "model_state_dict"
        ]
        agent.model.load_state_dict(state_dict)
        if config.load_temperature_states:
            temp_states = torch.load(config.checkpoint_path, map_location=device)[
                "temperature_state_dict"
            ]
            agent.load_temperature_from_state_dict(temp_states)
    # agent = accelerator.prepare(agent)

    if config.anchor_path is not None:
        agent.setup_anchor(
            anchor_checkpoint=config.anchor_path,
            map_location=device,
        )

    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=dict(config),
        )

    offpolicy_train_loop(
        env=train_env,
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        decode_f=decode_f,
        **config,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="archerTest")
    parser.add_argument("--model_type", type=str, default="small")
    parser.add_argument("--alg_name", type=str, default="archer")
    parser.add_argument("--logs_dir", type=str, default="./logs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port_number", type=int, default=1000)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_temperature", type=float, default=0.1)
    parser.add_argument("--temp_start", type=float, default=1.0)
    parser.add_argument("--temp_end", type=float, default=1.0)
    parser.add_argument("--temp_steps", type=int, default=1)
    parser.add_argument("--warmup_iter", type=int, default=20)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--num_eval_envs", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--env_type", type=str, default="easy")

    args = parser.parse_args()

    # Initialize Hydra
    hydra.initialize(config_path=f"./config/{args.alg_name}/", version_base=None)
    config = hydra.compose(config_name="aws")
    from datetime import datetime

    model = args.model
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = (
        args.logs_dir
        + "/"
        + args.project_name
        + "/"
        + args.alg_name
        + "/"
        + model
        + "/"
        + args.env_type
        + "/"
        + dt_string
    )
    config.save_path = log_dir
    config.seed = args.seed
    config.port_number = args.port_number
    config.project_name = args.project_name + "_" + args.env_type + "_" + args.model_type
    config.run_name = args.alg_name + "_" + model + "_" + dt_string
    config.policy_lm = model
    config.kl_weight = args.kl_weight
    config.warmup_iter = args.warmup_iter
    config.temp_start = args.temp_start
    config.temp_end = args.temp_end
    config.temp_steps = args.temp_steps
    config.attn_implementation = args.attn_implementation
    config.eval_freq = args.eval_freq
    config.num_eval_envs = args.num_eval_envs
    config.num_envs = args.num_envs
    config.beam_size = args.beam_size
    config.eval_temperature = args.eval_temperature
    config.env_type = args.env_type

    main(config)
