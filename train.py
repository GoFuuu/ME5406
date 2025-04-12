"""
FetchReach-v1 Environment Training with Multiple Reinforcement Learning Algorithms
ME5406 Project
"""

import os
import time
import pickle
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import tianshou as ts
from tianshou.policy import SACPolicy, PPOPolicy, TD3Policy, DDPGPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic, ActorProb, Actor
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import FilterObservation, FlattenObservation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.python.summary.summary_iterator import summary_iterator
from os import listdir, makedirs
import argparse

# Utility functions
def save_dict_to_file(dict_obj, path, txt_name='hyperparameter_dict'):
    """Save dictionary to a text file"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{txt_name}.txt'), 'w') as f:
        f.write(str(dict_obj))

def learning_curve(episode_rewards, log_dir, window=10):
    """Generate learning curve from episode rewards"""
    # Calculate rolling window metrics
    rolling_average = episode_rewards.rolling(window=window, min_periods=window).mean().dropna()
    rolling_max = episode_rewards.rolling(window=window, min_periods=window).max().dropna()
    rolling_min = episode_rewards.rolling(window=window, min_periods=window).min().dropna()

    # Change column name
    rolling_average.columns = ['Average Reward']
    rolling_max.columns = ['Max Reward']
    rolling_min.columns = ['Min Reward']
    rolling_data = pd.concat([rolling_average, rolling_max, rolling_min], axis=1)

    # Plot
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=rolling_data)
    ax.fill_between(rolling_average.index, rolling_min.iloc[:, 0], rolling_max.iloc[:, 0], alpha=0.2)
    ax.set_title('Learning Curve')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    # Save figure
    plt.savefig(os.path.join(log_dir, f'learning_curve{window}.png'))
    plt.close()

def learning_curve_tianshou(log_dir, window=10):
    """Generate learning curve from Tianshou training logs"""
    # Find event file
    files = listdir(log_dir)
    for f in files:
        if 'events' in f:
            event_file = f
            break

    # Read episode rewards
    episode_rewards_list = []
    episode_rewards = pd.DataFrame(columns=['Reward'])
    try:
        for e in summary_iterator(os.path.join(log_dir, event_file)):
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/reward':
                    episode_rewards_list.append(e.summary.value[0].simple_value)
    except Exception as e:
        pass
    episode_rewards['Reward'] = episode_rewards_list

    # Learning curve
    learning_curve(episode_rewards, log_dir, window=window)

def plot_reward_distribution(rewards, log_dir):
    """Plot reward distribution"""
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, 'reward_dist_epoch1.png'))
    plt.close()

def create_sac_policy(env, device, model_hyperparameters):
    """Create a SAC policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=model_hyperparameters['learning_rate'])

    net_c2 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=model_hyperparameters['learning_rate'])

    # Alpha
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_lr = 1e-4
    alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        exploration_noise=None,
        estimation_step=model_hyperparameters['estimation_step'],
        action_space=env.action_space,
        alpha=alpha
    )

    return policy, actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, alpha_lr

def create_ppo_policy(env, device, model_hyperparameters):
    """Create a PPO policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critic
    net_c = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=model_hyperparameters['learning_rate'])

    # Create PPO policy
    dist = torch.distributions.Normal
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=actor_optim,
        dist_fn=dist,
        action_space=env.action_space,
        discount_factor=0.99,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        reward_normalization=True,
        dual_clip=None,
        value_clip=True,
        deterministic_eval=True,
        advantage_normalization=True,
        recompute_advantage=False
    )

    return policy, actor, actor_optim, critic, critic_optim

def create_td3_policy(env, device, model_hyperparameters):
    """Create a TD3 policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=model_hyperparameters['learning_rate'])

    net_c2 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=model_hyperparameters['learning_rate'])

    # Noise
    exploration_noise = GaussianNoise(sigma=0.1)
    policy_noise = 0.2
    noise_clip = 0.5
    update_actor_freq = 2

    policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        exploration_noise=exploration_noise,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        update_actor_freq=update_actor_freq,
        estimation_step=model_hyperparameters['estimation_step'],
        action_space=env.action_space
    )

    return policy, actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim

def create_ddpg_policy(env, device, model_hyperparameters):
    """Create a DDPG policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critic
    net_c = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True, device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=model_hyperparameters['learning_rate'])

    # Noise
    exploration_noise = GaussianNoise(sigma=0.1)

    policy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        exploration_noise=exploration_noise,
        estimation_step=model_hyperparameters['estimation_step'],
        action_space=env.action_space
    )

    return policy, actor, actor_optim, critic, critic_optim

def train_agent(algorithm, env_id, seed, device, load_pretrained_model=False, model_log_dir='', model_file='', buffer_file=''):
    """Train an agent using the specified algorithm"""
    start = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Logging directory
    model_name = f'Tianshou_{algorithm}'
    log_dir = 'logs/' + model_name + '_' + str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())) + '/'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    # Environment
    env = gym.make(env_id, render_mode=None)
    # In gymnasium, we need to specify which keys to filter
    observation_keys = ['observation', 'achieved_goal', 'desired_goal']
    env = FlattenObservation(FilterObservation(env, filter_keys=observation_keys))
    env.reset(seed=seed)
    num_envs = 1

    train_envs = ts.env.DummyVectorEnv(
        [lambda: FlattenObservation(FilterObservation(gym.make(env_id, render_mode=None), filter_keys=observation_keys)) for _ in range(num_envs)])
    test_envs = ts.env.DummyVectorEnv(
        [lambda: FlattenObservation(FilterObservation(gym.make(env_id, render_mode=None), filter_keys=observation_keys)) for _ in range(num_envs)])
    train_envs.reset(seed=seed)
    test_envs.reset(seed=seed)

    # Model hyperparameters
    model_hyperparameters = {'hidden_sizes': [128, 128], 'learning_rate': 1e-3, 'estimation_step': 1}

    # Create policy based on algorithm
    if algorithm == 'SAC':
        policy, actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, alpha_lr = create_sac_policy(env, device, model_hyperparameters)
    elif algorithm == 'PPO':
        policy, actor, actor_optim, critic, critic_optim = create_ppo_policy(env, device, model_hyperparameters)
    elif algorithm == 'TD3':
        policy, actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim = create_td3_policy(env, device, model_hyperparameters)
    elif algorithm == 'DDPG':
        policy, actor, actor_optim, critic, critic_optim = create_ddpg_policy(env, device, model_hyperparameters)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Collectors
    use_prioritised_replay_buffer = True
    prioritized_buffer_hyperparameters = {'total_size': 1_000_000, 'buffer_num': num_envs, 'alpha': 0.7, 'beta': 0.5}

    # For PPO, we use a different collector
    if algorithm == 'PPO':
        train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, num_envs))
        test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)
    else:  # For off-policy algorithms (SAC, TD3, DDPG)
        if use_prioritised_replay_buffer:
            train_collector = ts.data.Collector(policy, train_envs,
                                                ts.data.PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters),
                                                exploration_noise=True)
        else:
            train_collector = ts.data.Collector(policy, train_envs,
                                                ts.data.VectorReplayBuffer(
                                                    total_size=prioritized_buffer_hyperparameters['total_size'] * num_envs,
                                                    buffer_num=num_envs),
                                                exploration_noise=True)
        test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    # Test function
    def build_test_fn(num_episodes=8):
        def custom_test_fn(epoch, env_step):
            print(f"Epoch = {epoch}, Steps = {env_step}")

            # Save agent
            if algorithm == 'SAC':
                torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                            'critic1_optim': critic1_optim.state_dict(), 'critic2_optim': critic2_optim.state_dict()},
                           os.path.join(log_dir, f'{model_name}_epoch{epoch}.pth'))
            elif algorithm == 'PPO':
                torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                            'critic_optim': critic_optim.state_dict()},
                           os.path.join(log_dir, f'{model_name}_epoch{epoch}.pth'))
            elif algorithm == 'TD3':
                torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                            'critic1_optim': critic1_optim.state_dict(), 'critic2_optim': critic2_optim.state_dict()},
                           os.path.join(log_dir, f'{model_name}_epoch{epoch}.pth'))
            elif algorithm == 'DDPG':
                torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                            'critic_optim': critic_optim.state_dict()},
                           os.path.join(log_dir, f'{model_name}_epoch{epoch}.pth'))

            # Save buffer for off-policy algorithms
            if algorithm != 'PPO':
                pickle.dump(train_collector.buffer, open(os.path.join(log_dir, f'epoch{epoch}_train_buffer.pkl'), "wb"))

        return custom_test_fn

    # Training hyperparameters
    if algorithm == 'PPO':
        trainer_hyperparameters = {
            'max_epoch': 1,
            'step_per_epoch': 80_000,
            'step_per_collect': 2000,
            'repeat_per_collect': 10,
            'episode_per_test': 1,
            'batch_size': 64
        }
    else:  # For off-policy algorithms (SAC, TD3, DDPG)
        trainer_hyperparameters = {
            'max_epoch': 1,
            'step_per_epoch': 80_000,
            'step_per_collect': 10,
            'episode_per_test': 1,
            'batch_size': 64
        }

    # Save hyperparameters
    all_hyperparameters = dict(model_hyperparameters, **trainer_hyperparameters)
    if algorithm != 'PPO':
        all_hyperparameters.update(prioritized_buffer_hyperparameters)
        all_hyperparameters['use_prioritised_replay_buffer'] = use_prioritised_replay_buffer

    all_hyperparameters['seed'] = seed
    all_hyperparameters['algorithm'] = algorithm

    if algorithm == 'SAC':
        all_hyperparameters['alpha_lr'] = alpha_lr

    if load_pretrained_model:
        # Load model, optimisers and buffer
        checkpoint = torch.load(os.path.join(model_log_dir, model_file))
        policy.load_state_dict(checkpoint['model'])

        if algorithm == 'SAC' or algorithm == 'TD3':
            policy.actor_optim.load_state_dict(checkpoint['actor_optim'])
            policy.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
            policy.critic2_optim.load_state_dict(checkpoint['critic2_optim'])
        elif algorithm == 'PPO' or algorithm == 'DDPG':
            policy.actor_optim.load_state_dict(checkpoint['actor_optim'])
            policy.critic_optim.load_state_dict(checkpoint['critic_optim'])

        if algorithm != 'PPO' and buffer_file:
            train_collector.buffer = pickle.load(open(os.path.join(model_log_dir, buffer_file), "rb"))

        all_hyperparameters['load_pretrained_model'] = load_pretrained_model
        all_hyperparameters['model_log_dir'] = model_log_dir
        all_hyperparameters['model_file'] = model_file
        all_hyperparameters['buffer_file'] = buffer_file

    save_dict_to_file(all_hyperparameters, path=log_dir)

    # Start training
    if algorithm == 'PPO':
        result = ts.trainer.onpolicy_trainer(
            policy, train_collector, test_collector, **trainer_hyperparameters,
            train_fn=None, test_fn=build_test_fn(), stop_fn=None, logger=logger
        )
    else:  # For off-policy algorithms (SAC, TD3, DDPG)
        result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector, **trainer_hyperparameters,
            train_fn=None, test_fn=build_test_fn(), stop_fn=None, logger=logger
        )

    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=25)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    print(f"Results saved to {log_dir}")

    return log_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent on FetchReach environment")
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO", "TD3", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--env_id", type=str, default="FetchReach-v1",
                        help="Environment ID")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--load_pretrained", action="store_true",
                        help="Whether to load a pretrained model")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Directory containing the pretrained model")
    parser.add_argument("--model_file", type=str, default="",
                        help="Filename of the pretrained model")
    parser.add_argument("--buffer_file", type=str, default="",
                        help="Filename of the replay buffer")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    train_agent(
        algorithm=args.algorithm,
        env_id=args.env_id,
        seed=args.seed,
        device=device,
        load_pretrained_model=args.load_pretrained,
        model_log_dir=args.model_dir,
        model_file=args.model_file,
        buffer_file=args.buffer_file
    )
