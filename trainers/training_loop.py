import accelerate.utils as autils
from accelerate import Accelerator
from transformers import PreTrainedTokenizer
from envs.moto_cli_env import BatchedAWSEnv
from data_utils.utils import ReplayBuffer
import numpy as np
from tqdm import tqdm
from trainers.archer import OfflineArcherTrainer, ArcherTrainer
from agents import ArcherAgent, BCAgent, CHAIAgent
from trainers.filtered_bc import BCTrainer
from trainers.chai_bc.chai_bc_trainer import CHAIBCTrainer
import wandb
import os
import json
import torch
import time
from typing import Union, Optional
from envs.env_utils import batch_interact_environment


def offpolicy_train_loop(
    env: BatchedAWSEnv,
    eval_env: BatchedAWSEnv,
    agent: Union[ArcherAgent, BCAgent, CHAIAgent],
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
    warmup_iter: int = 20,
    rollout_size: int = 50,
    eval_size: int = 1,
    batch_size: int = 2,
    capacity: int = 500000,
    iterations: int = 10,
    epochs: int = 3,
    grad_accum_steps: int = 1,
    env_idx: Optional[int] = None,
    critic_lr: float = 1e-3,
    lm_lr: float = 1e-5,
    gamma: float = 0.9,
    tau: float = 0.1,
    quantile: float = 0.9,
    use_wandb: bool = False,
    actor_epochs: int = 3,
    max_grad_norm: float = 0.01,
    save_path: Optional[str] = None,
    save_freq: int = 25,
    eval_freq: int = 25,
    agent_type: str = "archer",
    decode_f: callable = lambda x: x,
    log_env_stats: bool = False,
    kl_weight: float = 0.1,
    kl_beta: float = 1.0,
    use_distribution_averaging: bool = False,
    sample_during_eval: bool = True,
    **kwargs,
):
    if (
        agent_type.lower() == "chai"
        or agent_type.lower() == "archer"
        or agent_type.lower() == "archer_llm"
    ):
        assert isinstance(agent, CHAIAgent) or isinstance(agent, ArcherAgent)
        trainer = ArcherTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
    elif agent_type.lower() == "online_filteredbc":
        assert isinstance(agent, BCAgent)
        trainer = BCTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            lm_lr=lm_lr,
            epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
    else:
        raise NotImplementedError
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    all_trajectories = []
    if accelerator.is_main_process:
        print("Creating new checkpoint directory")
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "train")
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(save_path, "eval")
        os.makedirs(filepath, exist_ok=True)
        if os.path.exists(os.path.join(save_path, "trainer.pt")):
            print("Loading trainer checkpoint")
            trainer.load(os.path.join(save_path, "trainer.pt"))
        if os.path.exists(os.path.join(save_path, "trajectories.pt")):
            print("Loading trajectories checkpoint")
            all_trajectories = torch.load(os.path.join(save_path, "trajectories.pt"))
        if os.path.exists(os.path.join(save_path, "replay_buffer.pt")):
            print("Loading replay buffer checkpoint")
            replay_buffer = torch.load(os.path.join(save_path, "replay_buffer.pt"))

    # agent.prepare()
    # main training loop
    # wait for all processes to finish
    accelerator.wait_for_everyone()
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # all the processes collect data in parallel
        info = {}
        do_eval = False
        trajectories = batch_interact_environment(
            agent=agent,
            env=env,
            num_trajectories=rollout_size,
            env_idx=env_idx,
            use_tqdm=False,
            decode_f=decode_f,
        )
        if log_env_stats:
            train_env_info = env.info()
            train_env_info["process_id"] = accelerator.process_index
        else:
            train_env_info = {"process_id": accelerator.process_index}
        # convert env to a list for gathering from distributed processes
        train_env_info = [train_env_info]
        if (i + 1) % eval_freq == 0:
            do_eval = True
            old_sample = agent.do_sample
            agent.do_sample = sample_during_eval
            agent.set_policy_to_eval_mode()
            eval_trajectories = batch_interact_environment(
                agent=agent,
                env=eval_env,
                num_trajectories=max(eval_size, eval_env.num_envs),
                env_idx=env_idx,
                use_tqdm=False,
                decode_f=decode_f,
            )
            agent.do_sample = old_sample
            agent.set_policy_to_train_mode()
            if log_env_stats:
                eval_env_info = eval_env.info()
                eval_env_info["process_id"] = accelerator.process_index
            else:
                eval_env_info = {"process_id": accelerator.process_index}
            # convert env to a list for gathering from distributed processes
            eval_env_info = [eval_env_info]
        # gather all the data from the interactions
        accelerator.wait_for_everyone()
        trajectories = autils.gather_object(trajectories)
        train_env_info = autils.gather_object(train_env_info)
        if do_eval:
            eval_trajectories = autils.gather_object(eval_trajectories)
            eval_env_info = autils.gather_object(eval_env_info)
        # perform logging and saving in the main process while the other processes wait below
        if accelerator.is_main_process:
            info.update(
                {
                    "rollout.mean": np.mean(
                        [d[0]["trajectory_reward"] for d in trajectories]
                    ),
                    "rollout.max": np.max(
                        [d[0]["trajectory_reward"] for d in trajectories]
                    ),
                    "rollout.min": np.min(
                        [d[0]["trajectory_reward"] for d in trajectories]
                    ),
                }
            )
            if do_eval:
                info.update(
                    {
                        "eval_rollout.mean": np.mean(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                        "eval_rollout.max": np.max(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                        "eval_rollout.min": np.min(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                    }
                )
            if use_wandb:
                info["iterations"] = i
                wandb.log(info)

            if log_env_stats:
                filename = f"train/env_info_iteration_{i}.json"
                filepath = os.path.join(save_path, filename)
                with open(filepath, "w") as f:
                    for inv in train_env_info:
                        json.dump(inv, f, indent=2)
                if do_eval:
                    filename = f"eval/env_info_iteration_{i}.json"
                    filepath = os.path.join(save_path, filename)
                    with open(filepath, "w") as f:
                        for inv in eval_env_info:
                            json.dump(inv, f, indent=2)

            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)
            info.update(
                {
                    "rollout.reward.mean": np.mean([d["reward"] for d in data]),
                    "rollout.reward.max": np.max([d["reward"] for d in data]),
                    "rollout.reward.min": np.min([d["reward"] for d in data]),
                }
            )
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, "replay_buffer.pt"))
            torch.save(all_trajectories, os.path.join(save_path, "trajectories.pt"))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, "trajectories.pt"))
        replay_buffer = torch.load(os.path.join(save_path, "replay_buffer.pt"))
        print("Training")
        if "filtered" in agent_type.lower():
            filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, quantile)
            # print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(
                filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories)
            )
            data = sum(filtered_trajectories, [])
            for d in data:
                filtered_buffer.insert(**d)
            train_info = trainer.update(
                filtered_buffer, no_update_actor=(i < warmup_iter)
            )
            train_info.update({"episode_reward_cutoff": cutoff})
        else:
            train_info = trainer.update(
                replay_buffer, no_update_actor=(i < warmup_iter)
            )
        train_info.update(
            {
                "agent_temperature": agent.temperature,
                "agent_eval_temperature": agent.eval_temperature,
            }
        )
        agent.update_temperature()
        if use_wandb and accelerator.is_main_process:
            train_info["iterations"] = i
            wandb.log(train_info)
        if (
            (i + 1) % save_freq == 0
            and save_path is not None
            and accelerator.is_main_process
        ):
            print("Saving")
            trainer.save(os.path.join(save_path, "trainer.pt"))
            torch.save(replay_buffer, os.path.join(save_path, "replay_buffer.pt"))


def offline_train_loop(
    eval_env: BatchedAWSEnv,
    agent: Union[ArcherAgent, BCAgent, CHAIAgent],
    replay_buffer: ReplayBuffer,
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
    eval_size: int = 1,
    warmup_iter: int = 20,
    iterations: int = 10,
    epochs: int = 3,
    grad_accum_steps: int = 1,
    critic_lr: float = 1e-3,
    lm_lr: float = 1e-5,
    gamma: float = 0.9,
    tau: float = 0.1,
    use_wandb: bool = False,
    actor_epochs: int = 3,
    max_grad_norm: float = 0.01,
    save_path: Optional[str] = None,
    save_freq: int = 25,
    eval_freq: int = 25,
    agent_type: str = "archer",
    decode_f: callable = lambda x: x,
    log_env_stats: bool = False,
    kl_weight: float = 0.0,
    kl_beta: float = 1.0,
    use_distribution_averaging: bool = False,
    sample_during_eval: bool = True,
    *args,
    **kwargs,
):
    no_update_actor = False
    if agent_type.lower() == "archer" or agent_type.lower() == "archer_llm":
        assert isinstance(agent, ArcherAgent)
        trainer = OfflineArcherTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
    elif agent_type.lower() == "bc" or agent_type.lower() == "offline_filteredbc":
        assert isinstance(agent, BCAgent)
        trainer = BCTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            lm_lr=lm_lr,
            epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
    elif agent_type.lower() == "chai_bc" or agent_type.lower() == "chai":
        assert isinstance(agent, CHAIAgent)
        trainer = CHAIBCTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            actor_epochs=actor_epochs,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            warmup_steps=0,
            kl_weight=kl_weight,
            kl_beta=kl_beta,
            use_distribution_averaging=use_distribution_averaging,
        )
        if agent_type.lower() == "chai":
            no_update_actor = True
    else:
        raise NotImplementedError

    if accelerator.is_main_process:
        print("Creating new checkpoint directory")
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "eval")
        os.makedirs(filepath, exist_ok=True)
        if os.path.exists(os.path.join(save_path, "trainer.pt")):
            # print("Not using existing checkpoint")
            print("Loading trainer from checkpoint")
            trainer.load(os.path.join(save_path, "trainer.pt"))
    # main training loop, wait for all the processes before proceeding.
    accelerator.wait_for_everyone()
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # if accelerator.is_main_process:
        # evaluate agent
        # accelerator.wait_for_everyone()
        info = {}
        # all the processes collect data in parallel during eval
        if (i + 1) % eval_freq == 0:
            old_sample = agent.do_sample
            agent.do_sample = sample_during_eval
            agent.set_policy_to_eval_mode()
            eval_trajectories = batch_interact_environment(
                agent=agent,
                env=eval_env,
                num_trajectories=max(eval_size, eval_env.num_envs),
                use_tqdm=False,
                decode_f=decode_f,
            )
            agent.set_policy_to_train_mode()
            agent.do_sample = old_sample
            if log_env_stats:
                env_info = eval_env.info()
                env_info["process_id"] = accelerator.process_index
            else:
                env_info = {"process_id": accelerator.process_index}
            env_info = [env_info]
            # once done with the data collection, we gather together the collected trajectories and infos
            accelerator.wait_for_everyone()
            eval_trajectories = autils.gather_object(eval_trajectories)
            env_info = autils.gather_object(env_info)
            # logging is done by the main process only
            if accelerator.is_main_process:
                info.update(
                    {
                        "eval_rollout.mean": np.mean(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                        "eval_rollout.max": np.max(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                        "eval_rollout.min": np.min(
                            [d[0]["trajectory_reward"] for d in eval_trajectories]
                        ),
                    }
                )
                if use_wandb:
                    info["iterations"] = i
                    wandb.log(info)
                if log_env_stats:
                    filename = f"eval/env_info_iteration_{i}.json"
                    filepath = os.path.join(save_path, filename)
                    with open(filepath, "w") as f:
                        for inv in env_info:
                            json.dump(inv, f, indent=2)
                # TODO: Implement action generation of the model on train data!
        # wait for all processes to finish and then update the agent
        accelerator.wait_for_everyone()
        not_update_actor = no_update_actor or (i < warmup_iter)
        train_info = trainer.update(replay_buffer, no_update_actor=not_update_actor)
        train_info.update(
            {
                "agent_temperature": agent.temperature,
                "agent_eval_temperature": agent.eval_temperature,
            }
        )
        agent.update_temperature()
        if use_wandb and accelerator.is_main_process:
            # TODO: Currently only the info from the main process is logged
            train_info["iterations"] = i
            wandb.log(train_info)
        if (
            (i + 1) % save_freq == 0
            and save_path is not None
            and accelerator.is_main_process
        ):
            print("Saving")
            trainer.save(os.path.join(save_path, f"trainer_iteration_{i}.pt"))
