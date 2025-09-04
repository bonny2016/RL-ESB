import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re

def read_results_file(filename):
    """Read the results from a text file"""
    with open(filename, 'r') as f:
        content = f.read()
    return content

def extract_rewards_from_terminal(filename):
    """Extract rewards from terminal output saved in a file"""
    rewards = []
    pattern = r"Episode \d+/\d+, Average Reward: ([-\d.]+)"
    
    with open(filename, 'r') as f:
        content = f.read()
        matches = re.finditer(pattern, content)
        for match in matches:
            reward = float(match.group(1))
            rewards.append(reward)
    return rewards

def plot_comparison():
    # Create figures directory if it doesn't exist
    if not os.path.exists('data/figures'):
        os.makedirs('data/figures')

    # Plot training rewards comparison
    plt.figure(figsize=(12, 6))
    
    # Extract rewards (tries both potential files for each algorithm)
    rewards_ddqn = []
    if os.path.exists('data/ddqn_training.txt'):
        rewards_ddqn = extract_rewards_from_terminal('data/ddqn_training.txt')
    else:
        print("Warning: DDQN training history not found, using evaluation results.")
        rewards_ddqn = extract_rewards_from_terminal('data/ddqn_results.txt')
    
    rewards_ppo = []
    if os.path.exists('data/ppo_training.txt'):
        rewards_ppo = extract_rewards_from_terminal('data/ppo_training.txt')
    else:
        print("Warning: PPO training history not found, using evaluation results.")
        rewards_ppo = extract_rewards_from_terminal('data/results.txt')
    
    rewards_reinforce = []
    if os.path.exists('data/reinforce_training.txt'):
        rewards_reinforce = extract_rewards_from_terminal('data/reinforce_training.txt')
    else:
        print("Warning: REINFORCE training history not found, using evaluation results.")
        rewards_reinforce = extract_rewards_from_terminal('data/reinforce_results.txt')
    
    rewards_a2c = []
    if os.path.exists('data/a2c_training.txt'):
        rewards_a2c = extract_rewards_from_terminal('data/a2c_training.txt')
    else:
        print("Warning: A2C training history not found, using evaluation results.")
        if os.path.exists('data/a2c_results.txt'):
            rewards_a2c = extract_rewards_from_terminal('data/a2c_results.txt')

    # Plot available data
    if rewards_ddqn:
        # Use actual number of data points * 10 (since we log every 10 episodes)
        max_episodes_ddqn = len(rewards_ddqn) * 10
        episodes_ddqn = np.linspace(10, max_episodes_ddqn, len(rewards_ddqn))
        plt.plot(episodes_ddqn, rewards_ddqn, label='DDQN', color='blue', alpha=0.7)
    
    if rewards_ppo:
        max_episodes_ppo = len(rewards_ppo) * 10
        episodes_ppo = np.linspace(10, max_episodes_ppo, len(rewards_ppo))
        plt.plot(episodes_ppo, rewards_ppo, label='PPO', color='red', alpha=0.7)
    
    if rewards_reinforce:
        max_episodes_reinforce = len(rewards_reinforce) * 10
        episodes_reinforce = np.linspace(10, max_episodes_reinforce, len(rewards_reinforce))
        plt.plot(episodes_reinforce, rewards_reinforce, label='REINFORCE', color='green', alpha=0.7)
    
    if rewards_a2c:
        max_episodes_a2c = len(rewards_a2c) * 10
        episodes_a2c = np.linspace(10, max_episodes_a2c, len(rewards_a2c))
        plt.plot(episodes_a2c, rewards_a2c, label='A2C', color='orange', alpha=0.7)

    plt.title('Training Rewards: DDQN vs PPO vs REINFORCE vs A2C')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the comparison plot
    plt.savefig('data/figures/algorithm_comparison.png')
    plt.close()

    # Print statistical comparison
    print("\nAlgorithm Performance Comparison:")
    print("---------------------------------")
    if rewards_ddqn:
        print(f"DDQN Final Average Reward: {rewards_ddqn[-1]:.2f}")
    else:
        print("DDQN results not available")
    
    if rewards_ppo:
        print(f"PPO Final Average Reward: {rewards_ppo[-1]:.2f}")
    else:
        print("PPO results not available")
    
    if rewards_reinforce:
        print(f"REINFORCE Final Average Reward: {rewards_reinforce[-1]:.2f}")
    else:
        print("REINFORCE results not available")
    
    if rewards_a2c:
        print(f"A2C Final Average Reward: {rewards_a2c[-1]:.2f}")
    else:
        print("A2C results not available")
    
    # Compare final bus usage if results files exist
    ddqn_results = ""
    ppo_results = ""
    reinforce_results = ""
    a2c_results = ""
    
    if os.path.exists('data/ddqn_results.txt'):
        ddqn_results = read_results_file('data/ddqn_results.txt')
    if os.path.exists('data/results.txt'):
        ppo_results = read_results_file('data/results.txt')
    if os.path.exists('data/reinforce_results.txt'):
        reinforce_results = read_results_file('data/reinforce_results.txt')
    if os.path.exists('data/a2c_results.txt'):
        a2c_results = read_results_file('data/a2c_results.txt')
    
    print("\nBus Usage Comparison:")
    print("--------------------")
    
    # Extract and print bus usage information from all results
    for results, algo_name in [(ddqn_results, 'DDQN'), (ppo_results, 'PPO'), (reinforce_results, 'REINFORCE'), (a2c_results, 'A2C')]:
        if results:
            lines = results.split('\n')
            found_bus_info = False
            for line in lines:
                if "Average Buses Used:" in line:
                    print(f"{algo_name}: {line.strip()}")
                    found_bus_info = True
                    break
            if not found_bus_info:
                print(f"{algo_name}: Bus usage information not found in results")
        else:
            print(f"{algo_name}: Results file not found")

if __name__ == "__main__":
    plot_comparison()
