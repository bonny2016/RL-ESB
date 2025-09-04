import numpy as np
import matplotlib.pyplot as plt
import time
from environment import BusSchedulingEnv
from ddqn_agent import DDQNAgent
import config
import os

def create_results_directory():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/figures'):
        os.makedirs('data/figures')

def plot_rewards(rewards, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Rewards over Episodes (DDQN)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f'data/figures/{filename}')
    plt.close()

def write_results(results, filename):
    with open(f'data/{filename}', 'w') as f:
        f.write(results)

def main():
    create_results_directory()
    
    # Start timing
    start_time = time.time()
    print(f"Starting DDQN training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    env = BusSchedulingEnv()
    
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    
    agent = DDQNAgent(state_dim, action_dim)
    
    # Training parameters
    n_episodes = 6000
    max_steps = 500
    
    # Training tracking
    episode_rewards = []
    best_reward = float('-inf')
    
    # Training loop
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            loss = agent.train()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update target network periodically
        if episode % agent.update_target_every == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('data/ddqn_best_model.pth')
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            progress = f'Episode {episode + 1}/{n_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}\n'
            print(progress, end='')
            with open('data/ddqn_training.txt', 'a') as f:
                f.write(progress)
    
    # Plot training rewards
    plot_rewards(episode_rewards, 'ddqn_training_rewards.png')
    
    # Evaluate best model
    agent.load('data/ddqn_best_model.pth')
    agent.epsilon = 0  # No exploration during evaluation
    
    eval_episodes = 10
    eval_rewards = []
    total_buses_used = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                eval_rewards.append(episode_reward)
                total_buses_used.append(env.get_total_buses_used())
                break
    
    # Write evaluation results
    results = f"""DDQN Evaluation Results:
Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}
Average Buses Used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}
Best Episode Reward: {max(eval_rewards):.2f}
Minimum Buses Used: {min(total_buses_used)}
"""
    write_results(results, 'ddqn_results.txt')
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"DDQN training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total DDQN training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save timing information
    with open('data/ddqn_training.txt', 'a') as f:
        f.write(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")

if __name__ == "__main__":
    main()
