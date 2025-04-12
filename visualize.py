"""
FetchReach-v1 Environment Visualization with Multiple Reinforcement Learning Algorithms
ME5406 Project
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic, Actor
from tianshou.policy import SACPolicy, PPOPolicy, TD3Policy, DDPGPolicy
from tianshou.exploration import GaussianNoise
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FilterObservation, FlattenObservation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

def generate_more_videos(env, actor, num_episodes=10, fps=30, resolution_scale=2, viz_dir=None, device='cpu'):
    """Generate multiple video samples of the trained agent"""
    print("\n=== Generating More Video Samples ===")
    video_dir = os.path.join(viz_dir, "more_videos")
    os.makedirs(video_dir, exist_ok=True)

    total_rewards = []
    success_count = 0

    for episode in range(num_episodes):
        print(f"Generating video {episode+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        frames = []
        step = 0

        # Record trajectory
        trajectory = []

        while not (done or truncated) and step < 50:
            # Get action
            obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = actor(obs_tensor)
                act = logits[0].cpu().numpy()

                # Ensure action shape is correct
                if len(act.shape) > 1:
                    act = act.flatten()

                # Ensure actions are within action space bounds
                act = np.clip(act, env.action_space.low, env.action_space.high)

            # Execute action
            obs_next, rew, done, truncated, info = env.step(act)
            episode_reward += rew

            # Record trajectory
            trajectory.append((obs, act, rew, obs_next))

            # Render and save frame
            frame = env.render()
            if isinstance(frame, list):
                frame = frame[0]

            # Increase resolution
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w * resolution_scale, h * resolution_scale), interpolation=cv2.INTER_CUBIC)

            # Add information text
            cv2.putText(frame, f"Episode: {episode+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * resolution_scale, (0, 0, 255), 2)
            cv2.putText(frame, f"Step: {step}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * resolution_scale, (0, 0, 255), 2)
            cv2.putText(frame, f"Reward: {rew:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * resolution_scale, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Reward: {episode_reward:.4f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * resolution_scale, (0, 0, 255), 2)

            # Add action information
            for i, a in enumerate(act):
                cv2.putText(frame, f"Action {i+1}: {a:.4f}", (10, 150 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * resolution_scale, (0, 255, 0), 2)

            # Convert color space
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

            # Update observation
            obs = obs_next
            step += 1

        # Determine if successful
        if episode_reward > -10:  # Judge success based on reward
            success_count += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward:.4f}, Steps: {step}")

        # Save video
        if frames:
            video_path = os.path.join(video_dir, f"episode_{episode+1}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in frames:
                video.write(frame)

            video.release()
            print(f"Video saved to {video_path}")

    success_rate = success_count / num_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")

    return total_rewards, success_rate

def collect_data(env, actor, num_episodes=100, device='cpu'):
    """Collect data for visualization"""
    print("\n=== Collecting Data for Visualization ===")
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    episode_rewards = []
    success_count = 0

    for episode in tqdm(range(num_episodes), desc="Collecting data"):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0

        while not (done or truncated) and step < 50:
            # Get action
            obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = actor(obs_tensor)
                act = logits[0].cpu().numpy()

                # Ensure action shape is correct
                if len(act.shape) > 1:
                    act = act.flatten()

                # Ensure actions are within action space bounds
                act = np.clip(act, env.action_space.low, env.action_space.high)

            # Execute action
            obs_next, rew, done, truncated, info = env.step(act)
            episode_reward += rew

            # Record data
            all_observations.append(obs)
            all_actions.append(act)
            all_rewards.append(rew)
            all_next_observations.append(obs_next)

            # Update observation
            obs = obs_next
            step += 1

        episode_rewards.append(episode_reward)

        # Determine if successful
        if episode_reward > -10:  # Judge success based on reward
            success_count += 1

    success_rate = success_count / num_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")

    return (np.array(all_observations), np.array(all_actions), np.array(all_rewards),
            np.array(all_next_observations), np.array(episode_rewards), success_rate)

def plot_reward_distribution(episode_rewards, viz_dir):
    """Plot reward distribution"""
    print("\n=== Plotting Reward Distribution ===")
    plt.figure(figsize=(10, 6))
    sns.histplot(episode_rewards, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'reward_distribution.png'))
    print(f"Reward distribution plot saved to {os.path.join(viz_dir, 'reward_distribution.png')}")

def plot_action_distribution(actions, action_shape, viz_dir):
    """Plot action distribution"""
    print("\n=== Plotting Action Distribution ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(action_shape[0]):
        sns.histplot(actions[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f'Action Dimension {i+1} Distribution')
        axes[i].set_xlabel('Action Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'action_distribution.png'))
    print(f"Action distribution plot saved to {os.path.join(viz_dir, 'action_distribution.png')}")

    # Plot action correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = np.corrcoef(actions.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Action Dimension Correlation')
    plt.savefig(os.path.join(viz_dir, 'action_correlation.png'))
    print(f"Action correlation heatmap saved to {os.path.join(viz_dir, 'action_correlation.png')}")

def plot_state_space(observations, viz_dir):
    """Plot state space visualization"""
    print("\n=== Plotting State Space Visualization ===")

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=2)
    obs_pca = pca.fit_transform(observations)

    plt.figure(figsize=(10, 8))
    plt.scatter(obs_pca[:, 0], obs_pca[:, 1], alpha=0.5, s=5)
    plt.title('State Space PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'state_space_pca.png'))
    print(f"State space PCA visualization saved to {os.path.join(viz_dir, 'state_space_pca.png')}")

    # Use t-SNE for dimensionality reduction
    try:
        tsne = TSNE(n_components=2, random_state=42)
        # If there's too much data, take a sample
        if len(observations) > 5000:
            sample_indices = np.random.choice(len(observations), 5000, replace=False)
            obs_sample = observations[sample_indices]
        else:
            obs_sample = observations

        obs_tsne = tsne.fit_transform(obs_sample)

        plt.figure(figsize=(10, 8))
        plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], alpha=0.5, s=5)
        plt.title('State Space t-SNE Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'state_space_tsne.png'))
        print(f"State space t-SNE visualization saved to {os.path.join(viz_dir, 'state_space_tsne.png')}")
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

def plot_reward_action_relationship(actions, rewards, action_shape, viz_dir):
    """Plot reward-action relationship"""
    print("\n=== Plotting Reward-Action Relationship ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(action_shape[0]):
        axes[i].scatter(actions[:, i], rewards, alpha=0.3, s=5)
        axes[i].set_title(f'Action Dimension {i+1} vs Reward')
        axes[i].set_xlabel(f'Action {i+1}')
        axes[i].set_ylabel('Reward')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'reward_action_relationship.png'))
    print(f"Reward-action relationship plot saved to {os.path.join(viz_dir, 'reward_action_relationship.png')}")

def plot_success_rate(success_rate, viz_dir):
    """Plot success rate statistics"""
    print("\n=== Plotting Success Rate Statistics ===")
    plt.figure(figsize=(8, 6))
    plt.bar(['Success', 'Failure'], [success_rate, 100 - success_rate])
    plt.title('Task Success Rate Statistics')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)

    # Add value labels
    plt.text(0, success_rate + 2, f'{success_rate:.1f}%', ha='center')
    plt.text(1, 100 - success_rate + 2, f'{100 - success_rate:.1f}%', ha='center')

    plt.savefig(os.path.join(viz_dir, 'success_rate.png'))
    print(f"Success rate statistics plot saved to {os.path.join(viz_dir, 'success_rate.png')}")

def create_visualization_html(video_rewards, success_rate, viz_dir):
    """Create HTML page to display all visualization results"""
    print("\n=== Creating Visualization Results HTML Page ===")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>FetchReach SAC Extended Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .section {{
            margin-bottom: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        @media (max-width: 992px) {{
            .video-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        @media (max-width: 576px) {{
            .video-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        video {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .stats {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .highlight {{
            color: #2c7be5;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>FetchReach SAC Extended Visualizations</h1>

    <div class="section">
        <h2>Overview</h2>
        <p>This page presents extended visualizations of the SAC algorithm in the FetchReach-v1 environment.</p>

        <div class="stats">
            <h3>Performance Statistics</h3>
            <p><span class="highlight">Average Reward:</span> {np.mean(video_rewards):.4f}</p>
            <p><span class="highlight">Reward Range:</span> [{np.min(video_rewards):.4f}, {np.max(video_rewards):.4f}]</p>
            <p><span class="highlight">Success Rate:</span> {success_rate:.2f}%</p>
        </div>
    </div>

    <div class="section">
        <h2>Video Samples</h2>
        <p>Here are 10 video samples of the trained agent performing the task:</p>

        <div class="video-grid">
"""

    # Add videos
    for i in range(1, min(11, len(video_rewards) + 1)):
        html_content += f"""
            <div>
                <h3>Sample {i}</h3>
                <video controls>
                    <source src="more_videos/episode_{i}.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p>Reward: {video_rewards[i-1]:.4f}</p>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Distribution Visualizations</h2>

        <div class="grid">
            <div>
                <h3>Reward Distribution</h3>
                <img src="reward_distribution.png" alt="Reward Distribution">
                <p>This plot shows the distribution of total rewards obtained over multiple runs.</p>
            </div>
            <div>
                <h3>Action Distribution</h3>
                <img src="action_distribution.png" alt="Action Distribution">
                <p>This plot shows the distribution of actions taken by the agent across each action dimension.</p>
            </div>
            <div>
                <h3>Action Correlation</h3>
                <img src="action_correlation.png" alt="Action Correlation">
                <p>This heatmap shows the correlation between different action dimensions.</p>
            </div>
            <div>
                <h3>Reward-Action Relationship</h3>
                <img src="reward_action_relationship.png" alt="Reward-Action Relationship">
                <p>This plot shows the relationship between each action dimension and the resulting reward.</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>State Space Visualization</h2>

        <div class="grid">
            <div>
                <h3>PCA Dimensionality Reduction</h3>
                <img src="state_space_pca.png" alt="State Space PCA Visualization">
                <p>Using Principal Component Analysis (PCA) to reduce the high-dimensional state space to 2D for visualization.</p>
            </div>
            <div>
                <h3>t-SNE Dimensionality Reduction</h3>
                <img src="state_space_tsne.png" alt="State Space t-SNE Visualization">
                <p>Using t-SNE algorithm to reduce the high-dimensional state space to 2D for visualization, better preserving local structure.</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Success Rate Statistics</h2>
        <img src="success_rate.png" alt="Success Rate Statistics">
        <p>This plot shows the success rate of the agent in completing the task.</p>
    </div>

    <footer>
        <p>Generated on {time.strftime('%Y-%m-%d')}</p>
    </footer>
</body>
</html>
"""

    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write(html_content)

    print(f"HTML page saved to {os.path.join(viz_dir, 'index.html')}")

def main(args):

    # Set result directory
    log_dir = args.log_dir
    model_path = args.model_file  # Use the full path directly
    viz_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Environment settings
    env_id = args.env_id
    observation_keys = ['observation', 'achieved_goal', 'desired_goal']
    seed = args.seed
    device = args.device

    # Create environment
    env = gym.make(env_id, render_mode='rgb_array')
    env = FilterObservation(env, filter_keys=observation_keys)
    env = FlattenObservation(env)
    env.reset(seed=seed)

    # Get environment information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")

    # Create policy based on algorithm
    if args.algorithm == 'SAC':
        policy, actor = create_sac_policy(env, device)
    elif args.algorithm == 'PPO':
        policy, actor = create_ppo_policy(env, device)
    elif args.algorithm == 'TD3':
        policy, actor = create_td3_policy(env, device)
    elif args.algorithm == 'DDPG':
        policy, actor = create_ddpg_policy(env, device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Load model parameters
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Model loaded from {model_path}")
    print(f"Model keys: {checkpoint.keys()}")
    policy.load_state_dict(checkpoint['model'])
    policy.eval()

    # 1. Generate more video samples
    video_rewards, video_success_rate = generate_more_videos(
        env, actor, num_episodes=args.num_videos, fps=30,
        resolution_scale=2, viz_dir=viz_dir, device=device
    )

    # 2. Collect data
    observations, actions, rewards, next_observations, episode_rewards, success_rate = collect_data(
        env, actor, num_episodes=args.num_episodes, device=device
    )

    # 3. Plot reward distribution
    plot_reward_distribution(episode_rewards, viz_dir)

    # 4. Plot action distribution
    plot_action_distribution(actions, env.action_space.shape, viz_dir)

    # 5. Plot state space visualization
    plot_state_space(observations, viz_dir)

    # 6. Plot reward-action relationship
    plot_reward_action_relationship(actions, rewards, env.action_space.shape, viz_dir)

    # 7. Plot success rate statistics
    plot_success_rate(success_rate, viz_dir)

    # 8. Create HTML page
    create_visualization_html(video_rewards, success_rate, viz_dir)

    print(f"\nAll visualization results have been saved to {viz_dir}")
    print(f"Please open {os.path.join(viz_dir, 'index.html')} to view the complete results")

def create_sac_policy(env, device):
    """Create a SAC policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=[128, 128], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    # Alpha
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        exploration_noise=None,
        estimation_step=1,
        action_space=env.action_space,
        alpha=alpha
    )

    return policy, actor

def create_ppo_policy(env, device):
    """Create a PPO policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=[128, 128], device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # Critic
    net_c = Net(state_shape, hidden_sizes=[128, 128], device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

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

    return policy, actor

def create_td3_policy(env, device):
    """Create a TD3 policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=[128, 128], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # Critics
    net_c1 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

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
        estimation_step=1,
        action_space=env.action_space
    )

    return policy, actor

def create_ddpg_policy(env, device):
    """Create a DDPG policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Actor
    net_a = Net(state_shape, hidden_sizes=[128, 128], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # Critic
    net_c = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Noise
    exploration_noise = GaussianNoise(sigma=0.1)

    policy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        exploration_noise=exploration_noise,
        estimation_step=1,
        action_space=env.action_space
    )

    return policy, actor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained agent on FetchReach environment")
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO", "TD3", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--log_dir", type=str, default="logs/Tianshou_SAC_12_Apr_2025_14_23_57/",
                        help="Directory containing the trained model")
    parser.add_argument("--model_file", type=str, default="logs/Tianshou_SAC_12_Apr_2025_22_55_02/Tianshou_SAC_epoch1.pth",
                        help="Full path to the trained model file")
    parser.add_argument("--env_id", type=str, default="FetchReach-v1",
                        help="Environment ID")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of videos to generate")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to collect data from")

    args = parser.parse_args()
    main(args)
