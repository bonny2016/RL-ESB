# run.py
"""
Main script for training and evaluating the PPO agent.
"""

import numpy as np
import time
from environment import BusSchedulingEnv
from ppo_agent import PPOAgent
from config import STATE_DIM, ACTION_DIM, NUM_EPISODES
from utils import create_results_directory, plot_rewards, write_results

def train():
    """
    Trains the PPO agent.

    Returns:
        tuple: A tuple containing:
            - list: A list of rewards for each episode.
            - list: A list of average rewards for each block of episodes.
            - PPOAgent: The trained PPO agent.
            - BusSchedulingEnv: The environment.
    """
    create_results_directory()
    # Start timing
    start_time = time.time()
    print(f"Starting PPO training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Clear previous training log
    with open('data/ppo_training.txt', 'w') as f:
        f.write('')  # Clear the file
        
    env = BusSchedulingEnv()
    env.print_problem()
    
    agent = PPOAgent(STATE_DIM, ACTION_DIM, hidden_size=64)
    episode_rewards = []
    avg_rewards = []  # average reward for each block of 50 episodes
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        trajectory = []  # list of (state, action, log_prob, reward, done)
        total_reward = 0
        
        while not done:
            action, log_prob, _ = agent.policy.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, log_prob, reward, done))
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        agent.update(trajectory)
        
        # Every 50 episodes, calculate the average reward over the last 50 episodes.
        if (episode + 1) % 10 == 0:  # Changed to match DDQN's reporting frequency
            avg_reward = np.mean(episode_rewards[-10:])  # Use last 10 episodes like DDQN
            avg_rewards.append(avg_reward)
            progress = f"Episode {episode+1}/{NUM_EPISODES}, Average Reward: {avg_reward:.2f}\n"
            print(progress, end='')
            with open('data/ppo_training.txt', 'a') as f:
                f.write(progress)
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"PPO training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total PPO training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save timing information
    with open('data/ppo_training.txt', 'a') as f:
        f.write(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    
    return episode_rewards, avg_rewards, agent, env

def evaluate(agent, env, num_episodes=10):
    """
    Evaluates the trained PPO agent.

    Args:
        agent (PPOAgent): The trained PPO agent.
        env (BusSchedulingEnv): The environment.
        num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
    """
    eval_rewards = []
    total_buses_used = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _, _ = agent.policy.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        eval_rewards.append(episode_reward)
        total_buses_used.append(env.get_total_buses_used())
    
    # Save evaluation results
    results = f"""PPO Evaluation Results:
    Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}
    Average Buses Used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}
    Best Episode Reward: {max(eval_rewards):.2f}
    Minimum Buses Used: {min(total_buses_used)}
    """
    write_results(results, 'results.txt')
    
    # Print final schedule for best episode
    env.print_solution('data/results.txt')

if __name__ == '__main__':
    rewards, avg_rewards, agent, env = train()
    
    plot_rewards(rewards, 'ppo_training_rewards.png', 'PPO Training Progress')
    
    evaluate(agent, env)
