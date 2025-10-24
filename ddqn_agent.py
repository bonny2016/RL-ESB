import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for the DDQN agent.

    This class defines a simple fully-connected neural network
    that takes a state as input and outputs Q-values for each action.

    Attributes:
        fc1 (nn.Linear): The first fully-connected layer.
        fc2 (nn.Linear): The second fully-connected layer.
        fc3 (nn.Linear): The output layer.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the DQNetwork.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
        """
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) agent.

    This class implements the DDQN algorithm, which uses two separate
    Q-networks (a Q-network and a target network) to stabilize learning.

    Attributes:
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        device (torch.device): The device (CPU or CUDA) to run the agent on.
        learning_rate (float): The learning rate for the optimizer.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        epsilon_min (float): The minimum value for epsilon.
        epsilon_decay (float): The decay rate for epsilon.
        batch_size (int): The size of the mini-batch for training.
        memory (deque): A replay buffer to store experiences.
        update_target_every (int): The frequency (in episodes) to update the target network.
        q_network (DQNetwork): The main Q-network.
        target_network (DQNetwork): The target Q-network.
        optimizer (torch.optim.Optimizer): The optimizer for the Q-network.
        criterion (nn.Module): The loss function.
        training_step (int): The number of training steps performed.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the DDQNAgent.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DDQN using device: {self.device}")
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.update_target_every = 10  # Update target network every N episodes
        
        # Networks
        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.training_step = 0
    
    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode has finished.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.
            valid_actions (list): A list of valid actions.

        Returns:
            int: The selected action.
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        
        # Mask invalid actions with large negative values
        mask = torch.ones(self.action_dim) * float('-inf')
        mask[valid_actions] = 0
        q_values = q_values + mask.to(self.device)
        
        return torch.argmax(q_values).item()
    
    def train(self):
        """
        Trains the Q-network using a mini-batch of experiences from the replay buffer.

        Returns:
            float: The loss value.
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first for better performance
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Convert numpy arrays to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using target network for value estimation (DDQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """
        Updates the target network with the weights of the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """
        Saves the model and optimizer state dictionaries.

        Args:
            path (str): The path to save the model to.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """
        Loads the model and optimizer state dictionaries.

        Args:
            path (str): The path to load the model from.
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
