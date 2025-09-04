# run.py
import numpy as np
import matplotlib.pyplot as plt 
from environment import BusSchedulingEnv
from ppo_agent import PPOAgent
from config import STATE_DIM, ACTION_DIM, NUM_EPISODES
import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist

# Custom dataset for trajectories
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
    def __len__(self):
        return len(self.trajectories)
    def __getitem__(self, idx):
        return self.trajectories[idx]

def train(rank, world_size):
    # Distributed setup
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    env = BusSchedulingEnv()
    env.print_problem()

    agent = PPOAgent(STATE_DIM, ACTION_DIM, hidden_size=64)
    episode_rewards = []
    avg_rewards = []
    all_trajectories = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        while not done:
            action, log_prob, _ = agent.policy.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, log_prob, reward, done))
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        all_trajectories.extend(trajectory)

        # Every 10 episodes, batch update using DataLoader
        if (episode + 1) % 10 == 0:
            dataset = TrajectoryDataset(all_trajectories)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)
            for batch in dataloader:
                agent.update(batch)
            all_trajectories = []

        # Every 50 episodes, calculate the average reward over the last 50 episodes.
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rewards.append(avg_reward)
            print(f"Episode {episode+1}/{NUM_EPISODES} - Average Reward: {avg_reward:.2f}")

    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()
    return episode_rewards, avg_rewards, agent, env

def evaluate(agent, env):
    state = env.reset()
    done = False
    while not done:
        action, _, _ = agent.policy.act(state)
        state, _, done, _ = env.step(action)
    env.print_solution()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
        # Note: For evaluation and plotting, you may want to run on rank 0 only or after training is complete
        # This is a simplified example; for full DDP, you may need to save/load model state
    else:
        rewards, avg_rewards, agent, env = train(0, 1)
        episodes = np.arange(1, NUM_EPISODES+1)
        plt.plot(episodes, rewards, label="Episode Reward", alpha=0.5)
        avg_episode_nums = np.arange(50, NUM_EPISODES+1, 50)
        plt.plot(avg_episode_nums, avg_rewards, label="Average Reward (per 50 eps)", color="red", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Progress over Training Episodes")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("data/figures/training_rewards.png")
        evaluate(agent, env)
