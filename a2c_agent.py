import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Actor-Critic network for A2C.

    This class defines a neural network with a shared body and two heads:
    one for the actor (policy) and one for the critic (value function).

    Attributes:
        device (torch.device): The device (CPU or CUDA) to run the model on.
        shared (nn.Sequential): The shared layers of the network.
        actor (nn.Sequential): The actor-specific layers.
        critic (nn.Linear): The critic-specific layer.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        """
        Initializes the ActorCritic network.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int, optional): The number of neurons in the hidden layers. Defaults to 64.
        """
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The action probabilities.
                - torch.Tensor: The state value.
        """
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
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
                - torch.Tensor: The estimated state value.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.forward(state)
        
        # Apply action masking if valid_actions are provided
        if valid_actions is not None:
            mask = torch.zeros_like(action_probs)
            mask[0, valid_actions] = 1
            action_probs = action_probs * mask
            action_probs = action_probs / action_probs.sum()  # Renormalize
        
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        return action.item(), log_prob, state_value

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.

    This class implements the A2C algorithm, which uses an actor-critic
    architecture to learn a policy and a value function simultaneously.

    Attributes:
        device (torch.device): The device (CPU or CUDA) to run the agent on.
        actor_critic (ActorCritic): The actor-critic network.
        optimizer (torch.optim.Optimizer): The optimizer for the network.
        gamma (float): The discount factor for future rewards.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=0.001, gamma=0.99):
        """
        Initializes the A2CAgent.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int, optional): The number of neurons in the hidden layers. Defaults to 64.
            lr (float, optional): The learning rate. Defaults to 0.001.
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C using device: {self.device}")
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        
    def update(self, states, actions, rewards, next_states, dones, log_probs):
        """
        Updates the actor-critic network.

        Args:
            states (list): A list of states.
            actions (list): A list of actions.
            rewards (list): A list of rewards.
            next_states (list): A list of next states.
            dones (list): A list of done flags.
            log_probs (list): A list of log probabilities of the actions.

        Returns:
            tuple: A tuple containing the total loss, actor loss, and critic loss.
        """
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        
        # Get state values
        _, state_values = self.actor_critic(states)
        _, next_state_values = self.actor_critic(next_states)
        
        state_values = state_values.squeeze()
        next_state_values = next_state_values.squeeze()
        
        # Calculate advantages using TD error
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function loss)
        critic_loss = advantages.pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), actor_loss.item(), critic_loss.item()
    
    def act(self, state, valid_actions):
        """
        Selects an action using the actor-critic network.

        Args:
            state (np.ndarray): The current state.
            valid_actions (list): A list of valid actions.

        Returns:
            tuple: The output of the actor_critic.act() method.
        """
        return self.actor_critic.act(state, valid_actions)
    
    def save(self, path):
        """
        Saves the model and optimizer state dictionaries.

        Args:
            path (str): The path to save the model to.
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Loads the model and optimizer state dictionaries.

        Args:
            path (str): The path to load the model from.
        """
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
