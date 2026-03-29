import matplotlib.pyplot as plt
import numpy as np
from q_learning import train_q_learning
from sarsa import train_sarsa

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_combined_evolution(env_name, total_epochs=30000):
    # Parameters from your slide
    alpha, gamma, epsilon = 0.1, 0.9, 0.1
    eval_freq = 200
    
    print(f"Training Q-Learning and SARSA on {env_name}...")
    
    # Run Q-Learning
    _, ql_train_raw, ql_eval = train_q_learning(env_name, alpha, gamma, epsilon, total_epochs, eval_freq)
    
    # Run SARSA
    _, sarsa_train_raw, sarsa_eval = train_sarsa(env_name, alpha, gamma, epsilon, total_epochs, eval_freq)

    # Setup Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Algorithm Comparison: {env_name}", fontsize=16)

    # --- Plot 1: Training Reward (Evolution) ---
    # We use a moving average to make the training "trend" visible
    ql_smooth = moving_average(ql_train_raw)
    sarsa_smooth = moving_average(sarsa_train_raw)
    
    ax[0].plot(ql_smooth, label="Q-Learning (Moving Avg)", alpha=0.8)
    ax[0].plot(sarsa_smooth, label="SARSA (Moving Avg)", alpha=0.8)
    ax[0].set_title("Training Reward per Epoch")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Reward")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # --- Plot 2: Evaluation Reward (The "Actual" Learning) ---
    ax[1].plot(ql_eval['epochs'], ql_eval['avg_rewards'], label="Q-Learning Eval")
    ax[1].plot(sarsa_eval['epochs'], sarsa_eval['avg_rewards'], label="SARSA Eval")
    ax[1].set_title(f"Evaluation (Avg Reward over 50 runs every {eval_freq} epochs)")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Average Reward")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can run this for either environment
    plot_combined_evolution("Taxi-v3", total_epochs=30000)
    plot_combined_evolution("FrozenLake-v1", total_epochs=30000)