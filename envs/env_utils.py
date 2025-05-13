from tqdm import tqdm
import numpy as np
from envs.moto_cli_env import BatchedAWSEnv
from typing import Callable, Optional


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory


def add_mc_return(trajectory, gamma=0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1])) * gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1) / gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards * gamma_matrix, axis=1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


def get_envs_for_training(
    num_train_envs: int,
    num_eval_envs: int,
    seed: int,
    port_number: int,
    env_kwarg: dict,
    num_processes: int,
    process_index: int,
):
    train_env, eval_env = None, None
    if num_train_envs > 0:
        # get the environment ids for this process
        start_index, end_index = get_start_and_end_index(
            total_length=num_train_envs,
            process_index=process_index,
            num_processes=num_processes,
        )
        num_envs = end_index - start_index
        # assign port numbers starting at port_number + env_id,
        # same for the seed --> each batch of environment will have
        # a different seed
        env_kwarg["port_number"] = port_number + start_index
        # print(f'end and start index train {end_index} {start_index} {process_index}')
        # print(f'train port number {env_kwarg["port_number"]}, process_id {process_index}')
        train_env = BatchedAWSEnv(
            seed=seed + start_index,
            num_envs=num_envs,
            env_kwargs=env_kwarg,
        )
        train_env.close()
    if num_eval_envs > 0:
        # repeat the process for eval envs
        start_index, end_index = get_start_and_end_index(
            total_length=num_eval_envs,
            process_index=process_index,
            num_processes=num_processes,
        )
        num_envs = end_index - start_index
        offset = num_train_envs + 121
        # add a random number on top to ensure that train and eval env do not overlap
        env_kwarg["port_number"] = port_number + start_index + offset
        print(f'eval port number { env_kwarg["port_number"]}, process_id {process_index}')
        eval_env = BatchedAWSEnv(
            seed=seed + start_index + offset,
            num_envs=num_envs,
            env_kwargs=env_kwarg,
        )
        eval_env.close()
    return train_env, eval_env


def get_start_and_end_index(
    total_length: int, process_index: int, num_processes: int = 1
):
    if num_processes == 1:
        return 0, total_length
    length = total_length
    # get number of samples per process
    num_samples_per_process, num_extras = divmod(length, num_processes)
    # start index is process index * num_samples_per_process
    start_index = process_index * num_samples_per_process + min(
        process_index, num_extras
    )

    end_index = (
        start_index + num_samples_per_process + (1 if process_index < num_extras else 0)
    )
    return start_index, end_index


def batch_interact_environment(
    agent,
    env: BatchedAWSEnv,
    num_trajectories: int,
    post_f: Callable = lambda x: x,
    use_tqdm: bool = True,
    decode_f: Callable = lambda x: x,
    env_idx: Optional[int] = None,
):
    """
    in a batched way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories // bsize), disable=not use_tqdm):
        done = False
        trajectories = [[] for _ in range(bsize)]
        batch_obs = env.reset(idx=env_idx)
        batch_done = [
            False,
        ] * bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action = agent.get_action(batch_obs)
            batch_return = env.step(decode_f(action))
            for i, result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result
                trajectories[i].append(
                    {
                        "observation": batch_obs[i],
                        "next_observation": next_obs,
                        "reward": r,
                        "done": done,
                        "action": action[i],
                    }
                )
                batch_obs[i] = next_obs
                batch_done[i] = done
            # obs = next_obs
        # print(trajectories[0][-1]["next_observation"])
        all_trajectories += [
            post_f(add_mc_return(add_trajectory_reward(trajectory)))
            for trajectory in trajectories
        ]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    # close environment before exiting the function, to tear down the AWS server
    env.close()
    return all_trajectories
