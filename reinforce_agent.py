import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
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
        return self.network(state)
    
    def act(self, state, valid_actions=None):
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
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"REINFORCE using device: {self.device}")
        self.policy = Policy(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
    def update(self, rewards, log_probs):
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
        return self.policy.act(state, valid_actions)
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
