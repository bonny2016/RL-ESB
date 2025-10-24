import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    """
    Policy network for the REINFORCE agent.

    This class defines a simple fully-connected neural network
    that takes a state as input and outputs a probability distribution
    over the actions.

    Attributes:
        device (torch.device): The device (CPU or CUDA) to run the model on.
        network (nn.Sequential): The neural network.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        """
        Initializes the Policy network.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int, optional): The number of neurons in the hidden layers. Defaults to 64.
        """
        super(Policy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The action probabilities.
        """
        return self.network(state)
    
    def act(self, state, valid_actions=None):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): The current state.
            valid_actions (list, optional): A list of valid actions. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - int: The selected action.
                - torch.Tensor: The log probability of the selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.forward(state)
        
        # Apply action masking if valid_actions are provided
        if valid_actions is not None:
            mask = torch.zeros_like(probs)
            mask[0, valid_actions] = 1
            probs = probs * mask
            probs = probs / probs.sum()  # Renormalize
        
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        return action.item(), log_prob

class REINFORCEAgent:
    """
    REINFORCE agent.

    This class implements the REINFORCE algorithm, which is a simple
    policy gradient method.

    Attributes:
        device (torch.device): The device (CPU or CUDA) to run the agent on.
        policy (Policy): The policy network.
        optimizer (torch.optim.Optimizer): The optimizer for the network.
        gamma (float): The discount factor for future rewards.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=0.001, gamma=0.99):
        """
        Initializes the REINFORCEAgent.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int, optional): The number of neurons in the hidden layers. Defaults to 64.
            lr (float, optional): The learning rate. Defaults to 0.001.
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"REINFORCE using device: {self.device}")
        self.policy = Policy(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
    def update(self, rewards, log_probs):
        """
        Updates the policy network.

        Args:
            rewards (list): A list of rewards for an episode.
            log_probs (list): A list of log probabilities of the actions taken.

        Returns:
            float: The policy loss.
        """
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
    
    def act(self, state, valid_actions):
        """
        Selects an action using the policy network.

        Args:
            state (np.ndarray): The current state.
            valid_actions (list): A list of valid actions.

        Returns:
            tuple: The output of the policy.act() method.
        """
        return self.policy.act(state, valid_actions)
    
    def save(self, path):
        """
        Saves the model and optimizer state dictionaries.

        Args:
            path (str): The path to save the model to.
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Loads the model and optimizer state dictionaries.

        Args:
            path (str): The path to load the model from.
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
