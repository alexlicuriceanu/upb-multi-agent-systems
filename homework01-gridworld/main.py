import os
import numpy as np
import matplotlib.pyplot as plt
from gridworld import Gridworld
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from double_q_learning import DoubleQLearningAgent

TRIALS = 10
OUTPUT_DIR = "./results/"

def run_trials(agent_class, env_kwargs, agent_kwargs, episodes=500, trials=5):
    num_agents = env_kwargs.get('num_agents', 1)
    all_trials_steps = []

    for trial in range(trials):
        env = Gridworld(**env_kwargs)
        agent_kwargs['num_agents'] = num_agents
        
        # Initialize 1 or 2 agents
        agents = [agent_class(env=env, **agent_kwargs) for _ in range(num_agents)]
        
        steps_per_episode = []
        for episode in range(episodes):
            state = env.reset()
            actions = [agents[i].choose_action(state, agent_idx=i) for i in range(num_agents)]
            
            done = False
            steps = 0
            
            while not done:
                # Pass single action or list of actions
                env_actions = actions[0] if num_agents == 1 else actions
                next_state, rewards, done = env.step(env_actions)
                
                # Unify rewards format
                if num_agents == 1: rewards = [rewards]
                    
                next_actions = [agents[i].choose_action(next_state, agent_idx=i) for i in range(num_agents)]
                
                for i in range(num_agents):
                    agents[i].learn(state, actions[i], rewards[i], next_state, next_actions[i], agent_idx=i)
                
                state = next_state
                actions = next_actions
                steps += 1
                
            steps_per_episode.append(steps)
        all_trials_steps.append(steps_per_episode)

    return np.array(all_trials_steps)

def smooth_data(data, window_size=20):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def plot_ablation(data_dict, title, grid_type, param_name="Alpha", group_by="parameter", window_size=20, num_agents=1):
    parameter_values = list(data_dict.keys())
    algorithms = ["Q-Learning", "SARSA", "Double Q-Learning"]
    
    if group_by == "parameter":
        outer_loop, inner_loop = parameter_values, algorithms
        suptitle = f"Convergence Time (Varying {param_name})"
    else:
        outer_loop, inner_loop = algorithms, parameter_values
        suptitle = f"{param_name} Variations"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, outer_val in enumerate(outer_loop):
        ax = axes[i]
        
        for j, inner_val in enumerate(inner_loop):
            parameter_value = outer_val if group_by == "parameter" else inner_val
            algorithm = inner_val if group_by == "parameter" else outer_val
            
            matrix = data_dict[parameter_value][algorithm]
            mean_steps = np.mean(matrix, axis=0)
            std_steps = np.std(matrix, axis=0)

            smoothed_mean = smooth_data(mean_steps, window_size)
            smoothed_std = smooth_data(std_steps, window_size)
            x_axis = np.arange(window_size - 1, len(mean_steps))

            label = algorithm if group_by == "parameter" else f"{param_name} = {parameter_value}"

            ax.plot(x_axis, smoothed_mean, label=label, linewidth=2)
            ax.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.15)

        ax.set_title(f"{param_name} = {outer_val}" if group_by == "parameter" else f"{outer_val}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episodes", fontsize=12)
        if i == 0: ax.set_ylabel(f"Steps to Goal (Window={window_size})", fontsize=12)
            
        ax.set_ylim(0, 150 if num_agents == 2 else 100) 
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle(suptitle, fontsize=16, y=1.05)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = "multi_" if num_agents == 2 else "single_"
    plt.savefig(os.path.join(OUTPUT_DIR, f"task1_{prefix}{group_by.lower()}_{param_name.lower()}_{grid_type}.png"), dpi=100)

def plot_comparison(results_dict, title="Performance Comparison", window_size=20, num_agents=1):
    plt.figure(figsize=(10, 6))
    
    for label_name, data_matrix in results_dict.items():
        mean_steps = np.mean(data_matrix, axis=0)
        std_steps = np.std(data_matrix, axis=0)
        
        smoothed_mean = smooth_data(mean_steps, window_size)
        smoothed_std = smooth_data(std_steps, window_size)
        x_axis = np.arange(window_size - 1, len(mean_steps))
        
        plt.plot(x_axis, smoothed_mean, label=label_name, linewidth=2)
        plt.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.15)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(f"Steps to Goal (Window={window_size})", fontsize=12)
    plt.ylim(0, 150 if num_agents == 2 else 100) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = "multi_" if num_agents == 2 else "single_"
    plt.savefig(os.path.join(OUTPUT_DIR, f"task2_{prefix}comparison.png"), dpi=100)

def run_task_1(num_agents=1, episodes=500, trials=10):    
    algorithms = {"Q-Learning": QLearningAgent, "SARSA": SarsaAgent, "Double Q-Learning": DoubleQLearningAgent}
    alphas, epsilons = [0.1, 0.5, 0.9], [0.01, 0.1, 0.4]
    baseline_alpha, baseline_epsilon = 0.5, 0.1

    mode = "Multi-Agent Task 1" if num_agents == 2 else "Single-Agent Task 1"

    for grid_type in ['A', 'B']:
        env_args = {'grid_type': grid_type, 'use_diagonals': False, 'num_agents': num_agents}

        print(f"{mode}: Alpha ablation for Gridworld {grid_type}")
        alpha_data = {a: {} for a in alphas}
        for alpha in alphas:
            for algo_name, agent_class in algorithms.items():
                alpha_data[alpha][algo_name] = run_trials(agent_class, env_args, {'alpha': alpha, 'epsilon': baseline_epsilon}, episodes, trials)

        print(f"{mode}: Epsilon ablation for Gridworld {grid_type}")
        epsilon_data = {e: {} for e in epsilons}
        for eps in epsilons:
            for algo_name, agent_class in algorithms.items():
                epsilon_data[eps][algo_name] = run_trials(agent_class, env_args, {'alpha': baseline_alpha, 'epsilon': eps}, episodes, trials)

        plot_ablation(epsilon_data, f"{mode}: Epsilon Ablation (Gridworld {grid_type})", grid_type, "Epsilon", "parameter", num_agents=num_agents)
        plot_ablation(alpha_data, f"{mode}: Alpha Ablation (Gridworld {grid_type})", grid_type, "Alpha", "parameter", num_agents=num_agents)
        plot_ablation(alpha_data, f"{mode}: Alpha vs Algorithm (Gridworld {grid_type})", grid_type, "Alpha", "algorithm", num_agents=num_agents)
        plot_ablation(epsilon_data, f"{mode}: Epsilon vs Algorithm (Gridworld {grid_type})", grid_type, "Epsilon", "algorithm", num_agents=num_agents)

def run_task_2(num_agents=1, episodes=500, trials=10):
    mode = "Multi-Agent Task 2" if num_agents == 2 else "Single-Agent Task 2"
    print(f"{mode}: SARSA 4 vs 8 moves")
    
    agent_kwargs = {'alpha': 0.5, 'epsilon': 0.1}
    sarsa_4 = run_trials(SarsaAgent, {'grid_type': 'B', 'use_diagonals': False, 'num_agents': num_agents}, agent_kwargs, episodes, trials)
    sarsa_8 = run_trials(SarsaAgent, {'grid_type': 'B', 'use_diagonals': True, 'num_agents': num_agents}, agent_kwargs, episodes, trials)

    plot_comparison({"SARSA (4 Actions)": sarsa_4, "SARSA (8 Actions)": sarsa_8}, f"{mode}: SARSA Action Space Comparison", num_agents=num_agents)

if __name__ == "__main__":
    # Run single agent tasks
    run_task_1(num_agents=1, episodes=500, trials=10)
    run_task_2(num_agents=1, episodes=500, trials=10)

    # Run multi-agent tasks
    run_task_1(num_agents=2, episodes=5000, trials=10)
    run_task_2(num_agents=2, episodes=5000, trials=10)