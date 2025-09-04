# run.py
import numpy as np
import matplotlib.pyplot as plt 
import os
from environment import BusSchedulingEnv
from ppo_agent import PPOAgent
from config import STATE_DIM, ACTION_DIM, NUM_EPISODES

def train():
    # Clear previous training log
    if not os.path.exists('data'):
        os.makedirs('data')
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
    
    return episode_rewards, avg_rewards, agent, env

def evaluate(agent, env, num_episodes=10):
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
    with open('data/results.txt', 'w') as f:
        f.write(results)
    
    # Print final schedule for best episode
    env.print_solution('data/results.txt')

if __name__ == '__main__':
    rewards, avg_rewards, agent, env = train()
    
    episodes = np.arange(1, NUM_EPISODES+1)
    
    # Plot the reward for each episode
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.5)
    
    # Compute x-values for average rewards: these occur every 10 episodes.
    avg_episode_nums = np.arange(10, NUM_EPISODES+1, 10)  # Changed from 50 to 10
    plt.plot(avg_episode_nums, avg_rewards, label="Average Reward (per 10 eps)", color="red", linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Progress")  # Updated title
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("data/figures/ppo_training_rewards.png")  # Updated filename
    
    evaluate(agent, env)
