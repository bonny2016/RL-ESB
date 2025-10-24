"""
Main script for training and evaluating the A2C agent.
"""

import numpy as np
import time
from environment import BusSchedulingEnv
from a2c_agent import A2CAgent
from utils import create_results_directory, plot_rewards, write_results

# Hyperparameters
NUM_EPISODES = 6000
MAX_STEPS = 500
EVAL_EPISODES = 10

def main():
    """
    Main function for training and evaluating the A2C agent.
    """
    create_results_directory()
    
    # Start timing
    start_time = time.time()
    print(f"Starting A2C training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Clear previous training log
    with open('data/a2c_training.txt', 'w') as f:
        f.write('')
    
    env = BusSchedulingEnv()
    
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    
    agent = A2CAgent(state_dim, action_dim)
    
    # Training parameters
    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS
    
    # Training tracking
    episode_rewards = []
    best_reward = float('-inf')
    
    # Training loop
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Episode data collection
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        
        for step in range(max_steps):
            valid_actions = env.get_valid_actions()
            action, log_prob, state_value = agent.act(state, valid_actions)
            
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update agent with collected episode data
        if len(states) > 0:
            total_loss, actor_loss, critic_loss = agent.update(
                states, actions, rewards, next_states, dones, log_probs
            )
        
        episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('data/a2c_best_model.pth')
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            progress = f'Episode {episode + 1}/{n_episodes}, Average Reward: {avg_reward:.2f}\n'
            print(progress, end='')
            with open('data/a2c_training.txt', 'a') as f:
                f.write(progress)
    
    # Plot training rewards
    plot_rewards(episode_rewards, 'a2c_training_rewards.png', 'Training Rewards over Episodes (A2C)')
    
    # Evaluate best model
    agent.load('data/a2c_best_model.pth')
    agent.actor_critic.eval()
    
    eval_episodes = EVAL_EPISODES
    eval_rewards = []
    total_buses_used = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            action, log_prob, state_value = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                eval_rewards.append(episode_reward)
                total_buses_used.append(env.get_total_buses_used())
                break
    
    # Write evaluation results
    results = f"""A2C Evaluation Results:
Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}
Average Buses Used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}
Best Episode Reward: {max(eval_rewards):.2f}
Minimum Buses Used: {min(total_buses_used)}
"""
    write_results(results, 'a2c_results.txt')
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"A2C training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total A2C training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save timing information
    with open('data/a2c_training.txt', 'a') as f:
        f.write(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    
    print(f"\nA2C Training completed!")
    print(f"Best training reward: {best_reward:.2f}")
    print(f"Average evaluation reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average buses used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}")

if __name__ == "__main__":
    main()
