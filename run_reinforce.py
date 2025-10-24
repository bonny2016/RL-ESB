"""
Main script for training and evaluating the REINFORCE agent.
"""

import numpy as np
import time
from environment import BusSchedulingEnv
from reinforce_agent import REINFORCEAgent
from utils import create_results_directory, plot_rewards, write_results

# Hyperparameters
NUM_EPISODES = 6000
MAX_STEPS = 500
EVAL_EPISODES = 10

def main():
    """
    Main function for training and evaluating the REINFORCE agent.
    """
    create_results_directory()

    # Start timing
    start_time = time.time()
    print(f"Starting REINFORCE training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Clear previous training log
    with open('data/reinforce_training.txt', 'w') as f:
        f.write('')

    env = BusSchedulingEnv()
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    agent = REINFORCEAgent(state_dim, action_dim)

    episode_rewards = []
    best_reward = float('-inf')

    for episode in range(NUM_EPISODES):
        state = env.reset()
        rewards = []
        log_probs = []
        total_reward = 0
        for step in range(MAX_STEPS):
            valid_actions = env.get_valid_actions()
            action, log_prob = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            total_reward += reward
            state = next_state
            if done:
                break
        loss = agent.update(rewards, log_probs)
        episode_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('data/reinforce_best_model.pth')
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            progress = f'Episode {episode + 1}/{NUM_EPISODES}, Average Reward: {avg_reward:.2f}\n'
            print(progress, end='')
            with open('data/reinforce_training.txt', 'a') as f:
                f.write(progress)
    plot_rewards(episode_rewards, 'reinforce_training_rewards.png', 'Training Rewards over Episodes (REINFORCE)')

    # Evaluation
    agent.load('data/reinforce_best_model.pth')
    agent.policy.eval()
    eval_rewards = []
    total_buses_used = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        total_reward = 0
        while True:
            valid_actions = env.get_valid_actions()
            action, _ = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                eval_rewards.append(total_reward)
                total_buses_used.append(env.get_total_buses_used())
                break
    results = f"""REINFORCE Evaluation Results:\nAverage Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}\nAverage Buses Used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}\nBest Episode Reward: {max(eval_rewards):.2f}\nMinimum Buses Used: {min(total_buses_used)}\n"""
    write_results(results, 'reinforce_results.txt')
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"REINFORCE training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total REINFORCE training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save timing information
    with open('data/reinforce_training.txt', 'a') as f:
        f.write(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")

if __name__ == "__main__":
    main()
