import torch
import transformers
from envs.env_utils import get_envs_for_training
from agents import BCAgent
from trainers.training_loop import offline_train_loop
from utils import colorful_print
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import os
import numpy as np
from transformers import set_seed

transformers.logging.set_verbosity_error()


def main(config: "DictConfig"):
    # colorful_print(">>> Configuration file: " + CONFIG_NAME + "<<<", fg="blue")
    colorful_print(OmegaConf.to_yaml(config), fg="red")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_seed(config.seed)
    try:
        from huggingface_hub import login

        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(72000)))
    device = accelerator.device

    # load environment
    os.makedirs(config.save_path, exist_ok=True)
    if config.env_name == "aws_env":
        # if accelerator.is_main_process:
        # env = BatchedAWSEnv(seed=config.seed, num_envs=config.num_envs)
        # Each process will get a different environment with different port and seed
        _, eval_env = get_envs_for_training(
            num_train_envs=-1,
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
    if config.agent_type.lower() == "bc":
        print(">>> Using BC agent")
        agent = BCAgent(
            device=device,
            accelerator=accelerator,
            temperature=config.temperature,
            do_sample=config.do_sample,
            policy_lm=config.policy_lm,
            cache_dir=config.cache_dir,
            max_new_tokens=config.max_new_tokens,
            # eos_str=config.eos_str,
            max_length_tokenizer=config.max_length_tokenizer,
            use_lora=config.use_lora,
            use_bfloat16=config.use_bfloat16,
            eos_str=config.eos_str,
            quantization=config.quantization,
            use_better_transformer=config.use_better_transformer,
            attn_implementation=config.attn_implementation,
            eval_temperature=config.eval_temperature,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)[
            "model_state_dict"
        ]
        agent.model.load_state_dict(state_dict)
    # agent = accelerator.prepare(agent)

    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=dict(config),
        )
        wandb.define_metric("iterations")

    replay_buffer = torch.load(os.path.join(config.dataset_path, "replay_buffer.pt"))
    if config.add_eos_str_to_data:

        def add_eos_to_action(action):
            if action[-1] == config.eos_str:
                return action
            else:
                action = action + config.eos_str
            return action

        for i in range(replay_buffer.size):
            action = add_eos_to_action(replay_buffer.actions[i])
            replay_buffer.actions[i] = action

    offline_train_loop(
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        decode_f=decode_f,
        replay_buffer=replay_buffer,
        **config,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="BCTest")
    parser.add_argument("--alg_name", type=str, default="bc")
    parser.add_argument("--logs_dir", type=str, default="./logs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port_number", type=int, default=1000)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset_path", type=str, default='sft_dataset')
    parser.add_argument("--env_type", type=str, default="easy")
    parser.add_argument("--eval_temperature", type=float, default=0.1)
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--max_length_tokenizer", type=int, default=948)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--quantization", type=int, default=-1)
    args = parser.parse_args()

    # Initialize Hydra
    hydra.initialize(config_path=f"./config/{args.alg_name}/", version_base=None)
    config = hydra.compose(config_name="aws")
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = (
        args.logs_dir
        + "/"
        + args.project_name
        + "/"
        + args.alg_name
        + "/"
        + args.model
        + "/"
        + dt_string
    )
    config.save_path = log_dir
    config.seed = args.seed
    config.port_number = args.port_number
    config.project_name = args.project_name
    config.run_name = args.alg_name + "_" + args.model + "_" + dt_string
    config.policy_lm = args.model
    # by default we add the env type to the dataset path.
    config.dataset_path = args.dataset_path
    config.env_type = args.env_type
    config.eval_temperature = args.eval_temperature
    config.use_lora = bool(args.use_lora)
    config.max_length_tokenizer = args.max_length_tokenizer
    config.attn_implementation = args.attn_implementation
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    if args.quantization > 0:
        config.quantization = args.quantization

    #
    # config.policy_lm = "Qwen/Qwen2.5-1.5B-Instruct"
    # config.max_length_tokenizer = 2048
    main(config)