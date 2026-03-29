import matplotlib.pyplot as plt
import numpy as np
import os
from q_learning import train_q_learning
from sarsa import train_sarsa

def smooth_data(data, window_size=5):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_multiple_seeds(train_func, env_name, alpha, gamma, epsilon, epochs, eval_freq, eval_epochs, num_runs=5):
    all_rewards = []
    epochs_arr = None
    
    for run in range(num_runs):
        _, _, eval_metrics = train_func(env_name, alpha, gamma, epsilon, epochs, eval_freq, eval_epochs)
        all_rewards.append(eval_metrics['avg_rewards'])
        
        if epochs_arr is None:
            epochs_arr = eval_metrics['epochs']
            
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    
    return np.array(epochs_arr), mean_rewards

def plot_combined_parameter_variations(env_name, total_epochs=5000, eval_freq=100, eval_epochs=50, smoothing_window=5, num_runs=5):
    base_alpha, base_gamma, base_epsilon = 0.1, 0.9, 0.5
    param_grid = {'gamma': [0.5, 0.9], 'epsilon': [0.1, 0.5, 0.8], 'alpha': [0.1, 0.5, 0.9]}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for param_name, values in param_grid.items():
        plt.figure(figsize=(12, 6))
        
        for i, val in enumerate(values):
            a, g, e = base_alpha, base_gamma, base_epsilon
            if param_name == 'gamma': g = val
            elif param_name == 'epsilon': e = val
            elif param_name == 'alpha': a = val
            
            ql_epochs, ql_mean = run_multiple_seeds(train_q_learning, env_name, a, g, e, total_epochs, eval_freq, eval_epochs, num_runs)
            sarsa_epochs, sarsa_mean = run_multiple_seeds(train_sarsa, env_name, a, g, e, total_epochs, eval_freq, eval_epochs, num_runs)
            
            color = colors[i % len(colors)]
                        
            # Q-Learning plot
            if len(ql_mean) >= smoothing_window:
                smoothed_ql = smooth_data(ql_mean, smoothing_window)
                smooth_x = ql_epochs[(smoothing_window-1):]
                plt.plot(smooth_x, smoothed_ql, label=f'QL ({param_name}={val})', color=color, linestyle='-')

            # SARSA plot
            if len(sarsa_mean) >= smoothing_window:
                smoothed_sarsa = smooth_data(sarsa_mean, smoothing_window)
                smooth_x = sarsa_epochs[(smoothing_window-1):]
                plt.plot(smooth_x, smoothed_sarsa, label=f'SARSA ({param_name}={val})', color=color, linestyle='--')

        plt.title(f"{param_name.capitalize()} ablation ({env_name} - Avg over {num_runs} runs)")
        plt.xlabel("Training Epochs")
        plt.ylabel("Average Evaluation Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout() 
        
        filename = f"{env_name}_{param_name}_ablation.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

def plot_baseline_comparison(env_name, total_epochs=5000, eval_freq=50, eval_epochs=50, smoothing_window=5, num_runs=5):
    alpha, gamma, epsilon = 0.1, 0.9, 0.5
    
    ql_epochs, ql_mean = run_multiple_seeds(
        train_q_learning, env_name, alpha, gamma, epsilon, epochs=total_epochs, eval_freq=eval_freq, eval_epochs=eval_epochs, num_runs=num_runs
    )
    sarsa_epochs, sarsa_mean = run_multiple_seeds(
        train_sarsa, env_name, alpha, gamma, epsilon, epochs=total_epochs, eval_freq=eval_freq, eval_epochs=eval_epochs, num_runs=num_runs
    )
    
    plt.figure(figsize=(10, 6))
    
    if len(ql_mean) >= smoothing_window:
        smoothed_ql = smooth_data(ql_mean, smoothing_window)
        smooth_x = ql_epochs[(smoothing_window-1):]
        plt.plot(smooth_x, smoothed_ql, label='Q-Learning', color='tab:blue', linestyle='-', linewidth=2)

    if len(sarsa_mean) >= smoothing_window:
        smoothed_sarsa = smooth_data(sarsa_mean, smoothing_window)
        smooth_x = sarsa_epochs[(smoothing_window-1):]
        plt.plot(smooth_x, smoothed_sarsa, label='SARSA', color='tab:orange', linestyle='--', linewidth=2)

    plt.title(f"Comparison: Q-Learning vs SARSA ({env_name} - Avg over {num_runs} runs)")
    plt.xlabel("Training Epochs")
    plt.ylabel(f"Average Evaluation Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{env_name}_comparison.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":   
    # task 1
    plot_baseline_comparison("Taxi-v3", total_epochs=5000, eval_freq=100, eval_epochs=50, smoothing_window=5, num_runs=1)
    plot_baseline_comparison("FrozenLake-v1", total_epochs=20000, eval_freq=300, eval_epochs=50, smoothing_window=5, num_runs=1)

    # task 2
    plot_combined_parameter_variations("Taxi-v3", total_epochs=5000, eval_freq=100, eval_epochs=50, smoothing_window=5, num_runs=5)
    plot_combined_parameter_variations("FrozenLake-v1", total_epochs=20000, eval_freq=300, eval_epochs=50, smoothing_window=5, num_runs=5)