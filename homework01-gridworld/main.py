import os

import numpy as np
import matplotlib.pyplot as plt
from gridworld import Gridworld
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from double_q_learning import DoubleQLearningAgent

EPISODES = 500
TRIALS = 10
OUTPUT_DIR = "./results/"


def run_multiple_trials(agent_class, env_kwargs, agent_kwargs, episodes=500, trials=5):
    all_trials_steps = []

    for trial in range(trials):
        env = Gridworld(**env_kwargs)
        agent = agent_class(env=env, **agent_kwargs)

        steps = agent.train(episodes)
        all_trials_steps.append(steps)

    return np.array(all_trials_steps)

def smooth_data(data, window_size=20):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def plot_ablation(data_dict, title, grid_type, param_name="Alpha", group_by="parameter", window_size=20):
    parameter_values = list(data_dict.keys())
    algorithms = ["Q-Learning", "SARSA", "Double Q-Learning"]
    
    # Set up iteration based on grouping preference
    if group_by == "parameter":
        outer_loop = parameter_values
        inner_loop = algorithms
        suptitle = f"Convergence Time (Varying {param_name})"
    elif group_by == "algorithm":
        outer_loop = algorithms
        inner_loop = parameter_values
        suptitle = f"{param_name} Variations"
    else:
        raise ValueError("group_by must be either 'parameter' or 'algorithm'")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, outer_val in enumerate(outer_loop):
        ax = axes[i]
        
        for j, inner_val in enumerate(inner_loop):
            
            # Map the outer/inner values back to the specific dictionary keys
            parameter_value = outer_val if group_by == "parameter" else inner_val
            algorithm = inner_val if group_by == "parameter" else outer_val
            
            matrix = data_dict[parameter_value][algorithm]
            
            mean_steps = np.mean(matrix, axis=0)
            std_steps = np.std(matrix, axis=0)

            smoothed_mean = smooth_data(mean_steps, window_size)
            smoothed_std = smooth_data(std_steps, window_size)
            x_axis = np.arange(window_size - 1, len(mean_steps))

            # Determine labels based on grouping
            label = algorithm if group_by == "parameter" else f"{param_name} = {parameter_value}"

            ax.plot(x_axis, smoothed_mean, label=label, linewidth=2)
            ax.fill_between(
                x_axis, 
                smoothed_mean - smoothed_std, 
                smoothed_mean + smoothed_std, 
                alpha=0.15
            )

        # Set specific titles based on grouping
        if group_by == "parameter":
            ax.set_title(f"{param_name} = {outer_val}", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{outer_val}", fontsize=14, fontweight='bold')

        ax.set_xlabel(f"Episodes", fontsize=12)

        if i == 0:
            ax.set_ylabel(f"Steps to Goal (Rolling Average, Window={window_size})", fontsize=12)
            
        ax.set_ylim(0, 100) 
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle(suptitle, fontsize=16, y=1.05)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"task1_{group_by.lower()}_{param_name.lower()}_{grid_type}.png"

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=400)


def plot_comparison(results_dict, title="Performance Comparison", window_size=20):
    plt.figure(figsize=(10, 6))
    
    for i, (label_name, data_matrix) in enumerate(results_dict.items()):
        mean_steps = np.mean(data_matrix, axis=0)
        std_steps = np.std(data_matrix, axis=0)
        
        smoothed_mean = smooth_data(mean_steps, window_size)
        smoothed_std = smooth_data(std_steps, window_size)
        x_axis = np.arange(window_size - 1, len(mean_steps))
        
        plt.plot(x_axis, smoothed_mean, label=label_name, linewidth=2)
        plt.fill_between(
            x_axis,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            alpha=0.15
        )

    plt.title(title, fontsize=14, fontweight='bold')

    plt.xlabel(f"Episodes", fontsize=12)
    plt.ylabel(f"Steps to Goal (Rolling Average, Window={window_size})", fontsize=12)
    plt.ylim(0, 100) 

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.savefig(os.path.join(OUTPUT_DIR, f"task2_comparison.png"), dpi=400)


def run_task_1():    
    algorithms = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent,
        "Double Q-Learning": DoubleQLearningAgent
    }

    alphas = [0.1, 0.5, 0.9]
    epsilons = [0.01, 0.1, 0.4]
    baseline_alpha, baseline_epsilon = 0.5, 0.1

    for grid_type in ['A', 'B']:
       
        env_args = {'grid_type': grid_type, 'use_diagonals': False}

        # Alpha ablation
        print(f"Task 1: Alpha ablation for Gridworld {grid_type}")

        alpha_data = {}
        for alpha in alphas:
            alpha_data[alpha] = {}

            for algo_name, agent_class in algorithms.items():

                alpha_data[alpha][algo_name] = run_multiple_trials(
                    agent_class, env_args, {'alpha': alpha, 'epsilon': baseline_epsilon}, EPISODES, TRIALS
                )

        # Epsilon ablation
        print(f"Task 1: Epsilon ablation for Gridworld {grid_type}")

        epsilon_data = {}
        for eps in epsilons:
            epsilon_data[eps] = {}

            for algo_name, agent_class in algorithms.items():
                
                epsilon_data[eps][algo_name] = run_multiple_trials(
                    agent_class, env_args, {'alpha': baseline_alpha, 'epsilon': eps}, EPISODES, TRIALS
                )
        
        # Hyperparameter vs algorithm
        plot_ablation(
            epsilon_data,
            title=f"Task 1: Epsilon Ablation (Gridworld {grid_type})",
            grid_type=grid_type,
            param_name="Epsilon",
            group_by="parameter"
        )

        plot_ablation(
            alpha_data,
            title=f"Task 1: Alpha Ablation (Gridworld {grid_type})",
            grid_type=grid_type,
            param_name="Alpha",
            group_by="parameter"
        )

        # Hyperparameter vs hyperparameter
        plot_ablation(
            alpha_data,
            title=f"Task 1: Alpha vs Algorithm (Gridworld {grid_type})",
            grid_type=grid_type,
            param_name="Alpha",
            group_by="algorithm"
        )

        plot_ablation(
            epsilon_data,
            title=f"Task 1: Epsilon vs Algorithm (Gridworld {grid_type})",
            grid_type=grid_type,
            param_name="Epsilon",
            group_by="algorithm"
        )

def run_task_2():

    env_4_actions = {'grid_type': 'B', 'use_diagonals': False}
    env_8_actions = {'grid_type': 'B', 'use_diagonals': True}
    agent_kwargs = {'alpha': 0.5, 'epsilon': 0.1}

    print("Task 2: SARSA 4 moves")
    sarsa_4 = run_multiple_trials(SarsaAgent, env_4_actions, agent_kwargs, EPISODES, TRIALS)

    print("Task 2: SARSA 8 moves")
    sarsa_8 = run_multiple_trials(SarsaAgent, env_8_actions, agent_kwargs, EPISODES, TRIALS)

    results = {
        "SARSA (4 Actions)": sarsa_4,
        "SARSA (8 Actions)": sarsa_8
    }

    plot_comparison(results, title="Task 2: SARSA Action Space Comparison (Gridworld B)")


if __name__ == "__main__":
    run_task_1()
    run_task_2()