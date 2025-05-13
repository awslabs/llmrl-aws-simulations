import torch
import transformers
from agents import BCAgent, CHAIAgent, ArcherAgent
from agents.linear_scheduler import LinearScheduler
from trainers.training_loop import offline_train_loop
from utils import colorful_print
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
import os
from envs.env_utils import get_envs_for_training
import numpy as np
from data_utils.utils import ReplayBuffer

transformers.logging.set_verbosity_error()

CONFIG_NAME = "aws"


def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: " + CONFIG_NAME + "<<<", fg="blue")
    colorful_print(OmegaConf.to_yaml(config), fg="red")
    try:
        from huggingface_hub import login

        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")
    init_process_kwarg = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(kwargs_handlers=[init_process_kwarg, ddp_kwargs])
    device = accelerator.device
    temperature = LinearScheduler(
        start=config.temp_start, end=config.temp_end, total_steps=config.temp_steps
    )

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
            },
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
        )
    else:
        raise NotImplementedError("Environment not implemented.")
    decode_f = lambda x: x
    # load decision model
    if (
        config.agent_type.lower() == "bc"
        or config.agent_type.lower() == "offline_filteredbc"
    ):
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
        )

    elif config.agent_type.lower() == "chai_bc" or config.agent_type.lower() == "chai":
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
        )

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
        )
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)[
            "model_state_dict"
        ]
        agent.model.load_state_dict(state_dict)

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

    # setup replay buffer
    replay_buffer = torch.load(config.dataset_path)
    train_buffer = ReplayBuffer(
        batch_size=config.batch_size, capacity=replay_buffer.size
    )
    if "filtered" in config.agent_type.lower():
        episode_rewards = replay_buffer.mc_returns[: replay_buffer.size]
        cutoff = np.quantile(episode_rewards, config.quantile)
        indices = np.where(replay_buffer.mc_returns[: replay_buffer.size] >= cutoff)[
            0
        ].tolist()
        if config.use_wandb and accelerator.is_main_process:
            wandb.log(
                {
                    "cutoff": cutoff,
                }
            )
    else:
        indices = range(replay_buffer.size)
    for index in indices:
        train_buffer.insert(
            observation=replay_buffer.observations[index],
            action=replay_buffer.actions[index],
            reward=replay_buffer.rewards[index],
            next_observation=replay_buffer.next_observations[index],
            done=replay_buffer.dones[index],
            mc_return=replay_buffer.mc_returns[index],
        )
    print(f"Added {len(indices)} data points. Train buffer size: {len(train_buffer)}.")
    del replay_buffer

    offline_train_loop(
        eval_env=eval_env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        decode_f=decode_f,
        replay_buffer=train_buffer,
        **config,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="archerOfflineTest")
    parser.add_argument("--alg_name", type=str, default="filtered_bc")
    parser.add_argument("--logs_dir", type=str, default="./logs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port_number", type=int, default=1000)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--warmup_iter", type=int, default=0)
    parser.add_argument("--temp_start", type=float, default=0.75)
    parser.add_argument("--temp_end", type=float, default=0.25)
    parser.add_argument("--temp_steps", type=int, default=50)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
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
    config.kl_weight = args.kl_weight
    config.warmup_iter = args.warmup_iter
    config.temp_start = args.temp_start
    config.temp_end = args.temp_end
    config.temp_steps = args.temp_steps
    config.attn_implementation = args.attn_implementation

    main(config)
