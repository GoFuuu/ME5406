"""
FetchReach-v1 Environment Testing with Soft Actor-Critic (SAC)
ME5406 Project
"""

import os
import numpy as np
import torch
import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import SACPolicy
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FilterObservation, FlattenObservation
import cv2
import argparse

def test_policy(env, policy, num_episodes=5, record_video=True, use_random=False, video_dir=None, device='cpu'):
    """Test the policy on the environment and optionally record videos"""
    total_rewards = []

    for episode in range(num_episodes):
        print(f"Testing episode {episode+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        frames = []
        step = 0

        while not (done or truncated):
            # Get action
            if use_random:
                act = env.action_space.sample()
            else:
                # Fix batch data format issue
                obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)
                with torch.no_grad():  # No gradient calculation
                    # Directly use actor network to get action
                    logits, _ = policy.actor(obs_tensor)
                    # Ensure action dimension is correct
                    act = logits[0].cpu().numpy()
                    # Ensure action is within action space bounds
                    act = np.clip(act, env.action_space.low, env.action_space.high)

                # Ensure action shape is correct
                if len(act.shape) > 1:
                    act = act.flatten()

            # Execute action
            obs_next, rew, done, truncated, info = env.step(act)

            # Record reward
            episode_reward += rew

            # Record video frames
            if record_video:
                try:
                    frame = env.render()
                    if isinstance(frame, list):
                        frame = frame[0]
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Add text information
                    cv2.putText(frame, f"Step: {step}, Reward: {rew:.4f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Total Reward: {episode_reward:.4f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Failed to render frame: {e}")

            # Update observation
            obs = obs_next
            step += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward:.4f}, Steps: {step}")

        # Save video
        if record_video and frames and video_dir:
            os.makedirs(video_dir, exist_ok=True)
            policy_type = "random" if use_random else "trained"
            video_path = os.path.join(video_dir, f"{policy_type}_episode_{episode+1}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            for frame in frames:
                video.write(frame)

            video.release()
            print(f"Video saved to {video_path}")

    return total_rewards

def main(args):
    # Set result directory
    log_dir = args.log_dir
    model_path = args.model_file  # Use the full path directly
    video_dir = os.path.join(log_dir, "test_videos")
    os.makedirs(video_dir, exist_ok=True)

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
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {action_shape}")

    # Create network structure
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

    # Create policy
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

    # Load model parameters
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Model loaded from {model_path}")
        print(f"Model keys: {checkpoint.keys()}")

        # Load model parameters
        policy.load_state_dict(checkpoint['model'])
        print("Model parameters loaded successfully")
        policy.eval()  # Set to evaluation mode
        use_random_policy = False
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using random policy instead")
        use_random_policy = True

    # Run test - using trained policy
    if not use_random_policy:
        print("\n=== Testing with trained policy ===")
        trained_rewards = test_policy(env, policy, num_episodes=args.num_episodes,
                                     record_video=args.record_video, use_random=False,
                                     video_dir=video_dir, device=device)
        print(f"Average reward with trained policy: {np.mean(trained_rewards):.4f}")

    # Run test - using random policy for comparison
    if args.test_random:
        print("\n=== Testing with random policy ===")
        random_rewards = test_policy(env, policy, num_episodes=args.num_episodes,
                                    record_video=args.record_video, use_random=True,
                                    video_dir=video_dir, device=device)
        print(f"Average reward with random policy: {np.mean(random_rewards):.4f}")

    print(f"Test videos saved to {video_dir}")

    # If both policies were tested, print comparison results
    if not use_random_policy and args.test_random:
        print("\n=== Comparison ===")
        print(f"Trained policy average reward: {np.mean(trained_rewards):.4f}")
        print(f"Random policy average reward: {np.mean(random_rewards):.4f}")
        print(f"Improvement: {np.mean(trained_rewards) - np.mean(random_rewards):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained SAC agent on FetchReach environment")
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
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to test")
    parser.add_argument("--record_video", action="store_true", default=True,
                        help="Whether to record videos")
    parser.add_argument("--test_random", action="store_true", default=True,
                        help="Whether to test random policy for comparison")

    args = parser.parse_args()
    main(args)
