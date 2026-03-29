import matplotlib.pyplot as plt
import os
from q_learning import train_q_learning
from sarsa import train_sarsa

def plot_combined_parameter_variations(env_name, total_epochs=5000, eval_freq=100):
    base_alpha = 0.1
    base_gamma = 0.9
    base_epsilon = 0.1

    param_grid = {
        'gamma': [0.5, 0.9],
        'epsilon': [0.1, 0.5, 0.8],
        'alpha': [0.1, 0.5, 0.9]
    }

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for param_name, values in param_grid.items():
        plt.figure(figsize=(12, 6))

        for i, val in enumerate(values):
            a, g, e = base_alpha, base_gamma, base_epsilon
            
            if param_name == 'gamma': g = val
            elif param_name == 'epsilon': e = val
            elif param_name == 'alpha': a = val
            
            # Run agents
            _, _, ql_eval = train_q_learning(env_name, a, g, e, total_epochs, eval_freq)
            _, _, sarsa_eval = train_sarsa(env_name, a, g, e, total_epochs, eval_freq)
            
            color = colors[i % len(colors)]
            
            # Plot lines
            plt.plot(ql_eval['epochs'], ql_eval['avg_rewards'], 
                     label=f'QL ({param_name}={val})', color=color, linestyle='-', linewidth=2)
            plt.plot(sarsa_eval['epochs'], sarsa_eval['avg_rewards'], 
                     label=f'SARSA ({param_name}={val})', color=color, linestyle='--', linewidth=2, alpha=0.8)

        # Formatting
        plt.title(f"Influence of {param_name.capitalize()} ({env_name})")
        plt.xlabel("Training Epochs")
        plt.ylabel("Average Evaluation Reward")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout() 

        filename = f"{env_name}_{param_name}_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved graph: {filename}")
        
        plt.close()

if __name__ == "__main__":   
    plot_combined_parameter_variations("Taxi-v3", total_epochs=25000, eval_freq=200)
    plot_combined_parameter_variations("FrozenLake-v1", total_epochs=25000, eval_freq=200)