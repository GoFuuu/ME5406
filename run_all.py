"""
Run the complete FetchReach training, testing, and visualization pipeline
ME5406 Project
"""

import os
import argparse
import subprocess
import time

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        print(f"Error running {description}. Return code: {process.returncode}")
        return False

    return True

def main(args):
    # Create log directory with timestamp
    timestamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())
    log_dir = f"logs/Tianshou_SAC_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)

    # Step 1: Training
    if args.train:
        success = run_command(
            f"python train.py",
            "Training SAC agent"
        )
        if not success:
            print("Training failed. Exiting.")
            return

    # Get the most recent log directory if not training
    if not args.train:
        log_dirs = sorted([d for d in os.listdir("logs") if d.startswith("Tianshou_SAC_")], reverse=True)
        if not log_dirs:
            print("No log directories found. Please train a model first.")
            return
        log_dir = os.path.join("logs", log_dirs[0])

    # Step 2: Testing
    if args.test:
        model_path = os.path.join(log_dir, "Tianshou_SAC_epoch1.pth")
        success = run_command(
            f"python test.py --log_dir {log_dir} --model_file {model_path} --num_episodes {args.num_episodes}",
            "Testing trained agent"
        )
        if not success:
            print("Testing failed. Exiting.")
            return

    # Step 3: Visualization
    if args.visualize:
        model_path = os.path.join(log_dir, "Tianshou_SAC_epoch1.pth")
        success = run_command(
            f"python visualize.py --log_dir {log_dir} --model_file {model_path} --num_videos {args.num_videos} --num_episodes {args.num_episodes}",
            "Generating visualizations"
        )
        if not success:
            print("Visualization failed. Exiting.")
            return

    print(f"\n{'='*80}")
    print(f"All tasks completed successfully!")
    print(f"Results saved to: {log_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete FetchReach pipeline")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run testing")
    parser.add_argument("--visualize", action="store_true", help="Run visualization")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes for testing/visualization")
    parser.add_argument("--num_videos", type=int, default=5, help="Number of videos to generate")

    args = parser.parse_args()

    # If no specific tasks are selected, run all
    if not (args.train or args.test or args.visualize):
        args.train = True
        args.test = True
        args.visualize = True

    main(args)
