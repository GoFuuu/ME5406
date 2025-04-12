"""
FetchReach-v1 Environment Training with Soft Actor-Critic (SAC)
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
from tianshou.policy import SACPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic, ActorProb
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import FilterObservation, FlattenObservation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.python.summary.summary_iterator import summary_iterator
from os import listdir, makedirs

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

if __name__ == '__main__':
    start = time.perf_counter()
    env_id = "FetchReach-v1"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pretrained_model = False
    model_log_dir, model_file, buffer_file = '', '', ''

    # Logging directory
    model_name = 'Tianshou_SAC'
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

    # Neural networks and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    model_hyperparameters = {'hidden_sizes': [128, 128], 'learning_rate': 1e-3, 'estimation_step': 1}

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                 device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=model_hyperparameters['learning_rate'])
    net_c2 = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                 device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=model_hyperparameters['learning_rate'])

    # Alpha
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_lr = 1e-4
    alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, exploration_noise=None,
                       estimation_step=model_hyperparameters['estimation_step'], action_space=env.action_space,
                       alpha=alpha)

    # Collectors
    use_prioritised_replay_buffer = True
    prioritized_buffer_hyperparameters = {'total_size': 1_000_000, 'buffer_num': num_envs, 'alpha': 0.7, 'beta': 0.5}
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
    def build_test_fn(num_episodes):
        def custom_test_fn(epoch, env_step):
            print(f"Epoch = {epoch}, Steps = {env_step}")

            # Save agent
            torch.save({'model': policy.state_dict(), 'actor_optim': actor_optim.state_dict(),
                        'critic1_optim': critic1_optim.state_dict(), 'critic2_optim': critic2_optim.state_dict()},
                       os.path.join(log_dir, f'{model_name}_epoch{epoch}.pth'))
            pickle.dump(train_collector.buffer, open(os.path.join(log_dir, f'epoch{epoch}_train_buffer.pkl'), "wb"))

        return custom_test_fn

    # Training
    trainer_hyperparameters = {'max_epoch': 1, 'step_per_epoch': 80_000, 'step_per_collect': 10,
                               'episode_per_test': 1, 'batch_size': 64}
    all_hypeparameters = dict(model_hyperparameters, **trainer_hyperparameters, **prioritized_buffer_hyperparameters)
    all_hypeparameters['seed'] = seed
    all_hypeparameters['use_prioritised_replay_buffer'] = use_prioritised_replay_buffer
    all_hypeparameters['alpha_lr'] = alpha_lr
    if load_pretrained_model:
        # Load model, optimisers and buffer
        checkpoint = torch.load(os.path.join(model_log_dir, model_file))
        policy.load_state_dict(checkpoint['model'])
        policy.actor_optim.load_state_dict(checkpoint['actor_optim'])
        policy.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
        policy.critic2_optim.load_state_dict(checkpoint['critic2_optim'])
        train_collector.buffer = pickle.load(open(os.path.join(model_log_dir, buffer_file), "rb"))
        all_hypeparameters['load_pretrained_model'] = load_pretrained_model
        all_hypeparameters['model_log_dir'] = model_log_dir
        all_hypeparameters['model_file'] = model_file
        all_hypeparameters['buffer_file'] = buffer_file
    save_dict_to_file(all_hypeparameters, path=log_dir)

    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=None, test_fn=build_test_fn(num_episodes=8), stop_fn=None,
                                          logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # Learning Curve
    learning_curve_tianshou(log_dir=log_dir, window=25)

    # Execution Time
    end = time.perf_counter()  # tensorboard --logdir './logs'
    print(f"\nExecution time = {end - start:.2f} second(s)")
    print(f"Results saved to {log_dir}")
