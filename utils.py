"""
Utility functions for the RL bus scheduling project.
"""

import matplotlib.pyplot as plt
import os

def create_results_directory():
    """
    Creates the directories for storing results and figures if they don't exist.
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/figures'):
        os.makedirs('data/figures')

def plot_rewards(rewards, filename, title):
    """
    Plots the training rewards and saves the figure.

    Args:
        rewards (list): A list of rewards for each episode.
        filename (str): The filename to save the plot to.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f'data/figures/{filename}')
    plt.close()

def write_results(results, filename):
    """
    Writes the evaluation results to a file.

    Args:
        results (str): The results string to write.
        filename (str): The filename to write the results to.
    """
    with open(f'data/{filename}', 'w') as f:
        f.write(results)
