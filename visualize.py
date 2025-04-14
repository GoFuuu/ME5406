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

def generate_more_videos(env, actor, num_episodes=10, fps=30, resolution_scale=2, viz_dir=None, device='cpu', max_steps=50, success_threshold=-10):
    """
    Generate multiple video samples of the trained agent's performance.

    This function runs the trained agent in the environment for multiple episodes,
    records videos of the agent's behavior, and calculates performance metrics.

    Args:
        env (gym.Env): The environment to run the agent in (must support render() method)
        actor (torch.nn.Module): The trained actor network that outputs actions
        num_episodes (int): Number of episodes to record
        fps (int): Frames per second for the output videos
        resolution_scale (int): Factor by which to scale the rendered frames
        viz_dir (str): Directory to save visualization results
        device (str): Device to run the model on ('cpu' or 'cuda')
        max_steps (int): Maximum number of steps per episode
        success_threshold (float): Reward threshold above which an episode is considered successful

    Returns:
        tuple: (total_rewards, success_rate)
            - total_rewards (list): List of total rewards for each episode
            - success_rate (float): Percentage of successful episodes
    """
    # Create directory for videos if it doesn't exist
    print("\n=== Generating More Video Samples ===")
    video_dir = os.path.join(viz_dir, "more_videos")
    os.makedirs(video_dir, exist_ok=True)

    # Initialize metrics tracking
    total_rewards = []
    success_count = 0

    # Process each episode
    for episode in range(num_episodes):
        print(f"Generating video {episode+1}/{num_episodes}")

        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        frames = []
        step = 0

        # Record trajectory for potential further analysis
        trajectory = []

        # Run episode
        while not (done or truncated) and step < max_steps:
            # Convert observation to tensor and get action from policy
            obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)

            with torch.no_grad():  # No need to track gradients during inference
                logits, _ = actor(obs_tensor)
                act = logits[0].cpu().numpy()

            # Process action to ensure correct format
            if len(act.shape) > 1:  # Handle case where action is a 2D array
                act = act.flatten()

            # Ensure actions are within environment's action space bounds
            act = np.clip(act, env.action_space.low, env.action_space.high)

            # Execute action in environment
            obs_next, rew, done, truncated, info = env.step(act)
            episode_reward += rew

            # Store transition for potential further analysis
            trajectory.append((obs, act, rew, obs_next))

            # Render environment and process frame
            frame = env.render()
            if isinstance(frame, list):  # Handle case where render returns a list of frames
                frame = frame[0]

            # Increase resolution for better visualization
            h, w, c = frame.shape
            frame = cv2.resize(frame, (w * resolution_scale, h * resolution_scale),
                              interpolation=cv2.INTER_CUBIC)

            # Add informative text overlays
            # Episode and step information (red text)
            text_color = (0, 0, 255)  # BGR format (red)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 * resolution_scale
            thickness = 2
            cv2.putText(frame, f"Episode: {episode+1}", (10, 30), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Step: {step}", (10, 60), font, font_scale, text_color, thickness)

            # Reward information
            cv2.putText(frame, f"Reward: {rew:.4f}", (10, 90), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Total Reward: {episode_reward:.4f}", (10, 120), font, font_scale, text_color, thickness)

            # Action information (green text)
            action_color = (0, 255, 0)  # BGR format (green)
            action_font_scale = 0.6 * resolution_scale
            for i, a in enumerate(act):
                cv2.putText(frame, f"Action {i+1}: {a:.4f}", (10, 150 + i * 30), font,
                           action_font_scale, action_color, thickness)

            # Convert color space for video writing (RGB to BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

            # Update for next step
            obs = obs_next
            step += 1

        # Determine if episode was successful based on reward threshold
        if episode_reward > success_threshold:
            success_count += 1

        # Record and display episode results
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward:.4f}, Steps: {step}")

        # Save video if frames were collected
        if frames:
            video_path = os.path.join(video_dir, f"episode_{episode+1}.mp4")

            # Get dimensions from the first frame
            height, width, _ = frames[0].shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write all frames to video
            for frame in frames:
                video.write(frame)

            # Release resources
            video.release()
            print(f"Video saved to {video_path}")

    # Calculate and display overall performance
    success_rate = success_count / num_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")

    return total_rewards, success_rate

def collect_data(env, actor, num_episodes=100, device='cpu', max_steps=50, success_threshold=-10):
    """
    Collect data from agent-environment interactions for visualization and analysis.

    This function runs the trained agent in the environment for multiple episodes,
    collecting observations, actions, rewards, and transitions for later analysis.

    Args:
        env (gym.Env): The environment to run the agent in
        actor (torch.nn.Module): The trained actor network that outputs actions
        num_episodes (int): Number of episodes to collect data from
        device (str): Device to run the model on ('cpu' or 'cuda')
        max_steps (int): Maximum number of steps per episode
        success_threshold (float): Reward threshold above which an episode is considered successful

    Returns:
        tuple: (observations, actions, rewards, next_observations, episode_rewards, success_rate)
            - observations (np.ndarray): Array of observations
            - actions (np.ndarray): Array of actions taken
            - rewards (np.ndarray): Array of rewards received
            - next_observations (np.ndarray): Array of next observations
            - episode_rewards (np.ndarray): Array of total rewards for each episode
            - success_rate (float): Percentage of successful episodes
    """
    print("\n=== Collecting Data for Visualization ===")

    # Initialize data collection arrays
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    episode_rewards = []
    success_count = 0

    # Process each episode with progress bar
    for episode in tqdm(range(num_episodes), desc="Collecting data"):
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0

        # Run episode
        while not (done or truncated) and step < max_steps:
            # Convert observation to tensor and get action from policy
            obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)

            with torch.no_grad():  # No need to track gradients during inference
                logits, _ = actor(obs_tensor)
                act = logits[0].cpu().numpy()

            # Process action to ensure correct format
            if len(act.shape) > 1:  # Handle case where action is a 2D array
                act = act.flatten()

            # Ensure actions are within environment's action space bounds
            act = np.clip(act, env.action_space.low, env.action_space.high)

            # Execute action in environment
            obs_next, rew, done, truncated, info = env.step(act)
            episode_reward += rew

            # Store transition data for analysis
            all_observations.append(obs)
            all_actions.append(act)
            all_rewards.append(rew)
            all_next_observations.append(obs_next)

            # Update for next step
            obs = obs_next
            step += 1

        # Record episode total reward
        episode_rewards.append(episode_reward)

        # Determine if episode was successful based on reward threshold
        if episode_reward > success_threshold:
            success_count += 1

    # Calculate and display overall performance
    success_rate = success_count / num_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Total transitions collected: {len(all_observations)}")

    # Convert lists to numpy arrays for easier analysis
    return (np.array(all_observations), np.array(all_actions), np.array(all_rewards),
            np.array(all_next_observations), np.array(episode_rewards), success_rate)

def plot_reward_distribution(episode_rewards, viz_dir):
    """
    Plot the distribution of episode rewards.

    This function creates a histogram with kernel density estimation to visualize
    the distribution of total rewards across episodes, providing insights into
    the agent's performance consistency.

    Args:
        episode_rewards (np.ndarray or list): Array of total rewards from each episode
        viz_dir (str): Directory to save the visualization

    Returns:
        None: The function saves the plot to disk but doesn't return any value
    """
    print("\n=== Plotting Reward Distribution ===")

    # Create figure with appropriate size
    plt.figure(figsize=(10, 6))

    # Plot histogram with kernel density estimation
    sns.histplot(episode_rewards, kde=True, color='blue', alpha=0.7)

    # Add statistical information as text
    stats_text = f"Mean: {np.mean(episode_rewards):.2f}\nStd: {np.std(episode_rewards):.2f}"
    stats_text += f"\nMin: {np.min(episode_rewards):.2f}\nMax: {np.max(episode_rewards):.2f}"
    stats_text += f"\nMedian: {np.median(episode_rewards):.2f}"

    # Position text in the upper right corner
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    plt.title('Reward Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Total Episode Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')

    # Ensure layout is tight
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(viz_dir, 'reward_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Reward distribution plot saved to {output_path}")

def plot_action_distribution(actions, action_shape, viz_dir):
    """
    Plot the distribution of actions across each action dimension.

    This function creates a grid of histograms showing the distribution of actions
    taken by the agent in each dimension of the action space. This helps analyze
    the agent's action preferences and biases.

    Args:
        actions (np.ndarray): Array of actions taken by the agent, shape (n_steps, action_dim)
        action_shape (tuple): Shape of the action space
        viz_dir (str): Directory to save the visualization

    Returns:
        None: The function saves the plot to disk but doesn't return any value
    """
    print("\n=== Plotting Action Distribution ===")

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot distribution for each action dimension
    for i in range(action_shape[0]):
        # Extract actions for this dimension
        dimension_actions = actions[:, i]

        # Plot histogram with kernel density estimation
        sns.histplot(dimension_actions, kde=True, ax=axes[i], color=f'C{i}', alpha=0.7)

        # Add statistical information
        stats_text = f"Mean: {np.mean(dimension_actions):.3f}\nStd: {np.std(dimension_actions):.3f}"
        stats_text += f"\nMin: {np.min(dimension_actions):.3f}\nMax: {np.max(dimension_actions):.3f}"

        # Add text box with statistics
        axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set titles and labels
        axes[i].set_title(f'Action Dimension {i+1} Distribution', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Action Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)

        # Add grid for better readability
        axes[i].grid(True, alpha=0.3, linestyle='--')

        # Add vertical line at zero for reference
        axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for i in range(action_shape[0], len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle('Action Distributions by Dimension', fontsize=16, fontweight='bold', y=0.98)

    # Ensure layout is tight
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle

    # Save figure
    output_path = os.path.join(viz_dir, 'action_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Action distribution plot saved to {output_path}")

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
    """
    Plot the relationship between actions and rewards.

    This function creates scatter plots showing how different action dimensions
    correlate with received rewards. It helps identify which action dimensions
    have the strongest impact on performance and the optimal action values.

    Args:
        actions (np.ndarray): Array of actions taken by the agent, shape (n_steps, action_dim)
        rewards (np.ndarray): Array of rewards received, shape (n_steps,)
        action_shape (tuple): Shape of the action space
        viz_dir (str): Directory to save the visualization

    Returns:
        None: The function saves the plot to disk but doesn't return any value
    """
    print("\n=== Plotting Reward-Action Relationship ===")

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot relationship for each action dimension
    for i in range(action_shape[0]):
        # Extract actions for this dimension
        dimension_actions = actions[:, i]

        # Create scatter plot
        scatter = axes[i].scatter(dimension_actions, rewards, alpha=0.5, s=10, c=rewards,
                                 cmap='viridis', edgecolor='none')

        # Add trend line using polynomial fit
        z = np.polyfit(dimension_actions, rewards, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(dimension_actions)
        axes[i].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

        # Add correlation coefficient
        correlation = np.corrcoef(dimension_actions, rewards)[0, 1]
        corr_text = f"Correlation: {correlation:.3f}"
        slope_text = f"Slope: {z[0]:.3f}"

        # Add text box with correlation information
        axes[i].text(0.05, 0.95, corr_text + '\n' + slope_text, transform=axes[i].transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set titles and labels
        axes[i].set_title(f'Reward vs Action Dimension {i+1}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(f'Action {i+1} Value', fontsize=10)
        axes[i].set_ylabel('Reward', fontsize=10)

        # Add grid for better readability
        axes[i].grid(True, alpha=0.3, linestyle='--')

        # Add vertical line at zero for reference
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.3)

        # Add horizontal line at zero reward for reference
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Hide any unused subplots
    for i in range(action_shape[0], len(axes)):
        axes[i].set_visible(False)

    # Add colorbar
    if action_shape[0] > 0:  # Only add if we have at least one plot
        cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', pad=0.01)
        cbar.set_label('Reward Value', fontsize=10)

    # Add overall title
    fig.suptitle('Relationship Between Actions and Rewards', fontsize=16, fontweight='bold', y=0.98)

    # Ensure layout is tight
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle

    # Save figure
    output_path = os.path.join(viz_dir, 'reward_action_relationship.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Reward-action relationship plot saved to {output_path}")

def plot_success_rate(success_rate, viz_dir):
    """
    Plot the success rate statistics of the agent.

    This function creates a bar chart showing the success and failure rates
    of the agent on the task, providing a clear visualization of the agent's
    overall performance.

    Args:
        success_rate (float): The success rate as a percentage (0-100)
        viz_dir (str): Directory to save the visualization

    Returns:
        None: The function saves the plot to disk but doesn't return any value
    """
    print("\n=== Plotting Success Rate Statistics ===")

    # Create figure with appropriate size
    plt.figure(figsize=(10, 6))

    # Define colors for success and failure
    colors = ['#2ecc71', '#e74c3c']  # Green for success, red for failure

    # Create bar chart
    bars = plt.bar(['Success', 'Failure'], [success_rate, 100 - success_rate], color=colors, alpha=0.8)

    # Add title and labels
    plt.title('Task Success Rate Statistics', fontsize=16, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 110)  # Leave room for labels above bars

    # Add value labels above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        value = success_rate if i == 0 else 100 - success_rate
        plt.text(bar.get_x() + bar.get_width()/2, height + 3,
                f"{value:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add horizontal line at 50% for reference
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.3)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add a text box with interpretation
    if success_rate >= 90:
        interpretation = "Excellent performance!"
    elif success_rate >= 70:
        interpretation = "Good performance"
    elif success_rate >= 50:
        interpretation = "Moderate performance"
    else:
        interpretation = "Needs improvement"

    plt.text(0.5, 0.05, interpretation, transform=plt.gca().transAxes,
             ha='center', va='bottom', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ensure layout is tight
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(viz_dir, 'success_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Success rate statistics plot saved to {output_path}")

def create_visualization_html(video_rewards, success_rate, viz_dir):
    """
    Create an HTML page to display all visualization results in an organized manner.

    This function generates a comprehensive HTML report that includes all the
    visualizations, performance metrics, and video samples in a well-structured
    format for easy viewing and analysis.

    Args:
        video_rewards (list): List of rewards from the video generation episodes
        success_rate (float): The success rate as a percentage (0-100)
        viz_dir (str): Directory containing the visualization results

    Returns:
        str: Path to the generated HTML file
    """
    print("\n=== Creating Visualization Results HTML Page ===")

    # Calculate additional statistics for the report
    avg_reward = np.mean(video_rewards)
    std_reward = np.std(video_rewards)
    min_reward = np.min(video_rewards)
    max_reward = np.max(video_rewards)
    median_reward = np.median(video_rewards)

    # Get current timestamp for the report
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Create HTML content with improved styling and organization
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
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c7be5;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .highlight {{
            color: #2c7be5;
            font-weight: bold;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>FetchReach Reinforcement Learning Visualizations</h1>
    <p>Generated on {timestamp}</p>

    <div class="section">
        <h2>Performance Overview</h2>
        <p>This page presents comprehensive visualizations of the reinforcement learning algorithm in the FetchReach-v1 environment.</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{success_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Reward</div>
                <div class="stat-value">{avg_reward:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Median Reward</div>
                <div class="stat-value">{median_reward:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Reward Range</div>
                <div class="stat-value">[{min_reward:.2f}, {max_reward:.2f}]</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Video Samples</h2>
        <p>Below are video samples of the trained agent performing the task:</p>

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
    """
    Main function to run the visualization process.

    This function coordinates the entire visualization workflow, including:
    1. Setting up the environment and loading the trained model
    2. Generating video samples of the agent's behavior
    3. Collecting data for analysis
    4. Creating various visualizations
    5. Generating an HTML report

    Args:
        args: Command-line arguments containing configuration parameters

    Returns:
        None
    """
    print("\n=== Starting Visualization Process ===")

    # Set up directories
    log_dir = args.log_dir
    model_path = args.model_file
    viz_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Visualization results will be saved to: {viz_dir}")

    # Environment settings
    env_id = args.env_id
    observation_keys = ['observation', 'achieved_goal', 'desired_goal']
    seed = args.seed
    device = args.device

    # Create environment with rendering capability
    print(f"\nSetting up environment: {env_id}")
    env = gym.make(env_id, render_mode='rgb_array')
    env = FilterObservation(env, filter_keys=observation_keys)
    env = FlattenObservation(env)
    env.reset(seed=seed)

    # Display environment information
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
