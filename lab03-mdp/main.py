import gym
import numpy as np
import matplotlib.pyplot as plt

from standard import standard_value_iteration
from gauss_seidel import gauss_seidel_vi
from prioritized_sweeping import prioritized_sweeping_vi
from policy_iteration import policy_iteration

GAMMA = 0.9
EPSILON = 1e-3
MAX_ITERATIONS = 5e5

def get_standard_vi_norms(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    iteration_count = 0
    norms_history = []
    
    while iteration_count < max_iters:
        delta = 0
        V_old = np.copy(V)
        for s in range(nS):
            action_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V_old[next_state])
            
            best_action_value = np.max(action_values)
            V[s] = best_action_value
            iteration_count += 1
            
            current_norm = np.linalg.norm(V - V_star)
            norms_history.append(current_norm)
            
            delta = max(delta, np.abs(best_action_value - V_old[s]))
            if iteration_count >= max_iters:
                break
        if delta <= epsilon or current_norm <= epsilon:
            break
    return iteration_count, norms_history


def generate_plots(env_name, is_slippery=False):
    if env_name == "FrozenLake-v1":
        env = gym.make(env_name, map_name="8x8", is_slippery=is_slippery)
    else:
        env = gym.make(env_name)
        
    print(f"Running on {env_name}:")


    V_star, iters_vi = standard_value_iteration(env=env, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS)
    iters_std, norms_std = get_standard_vi_norms(env=env, V_star=V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS)
    print(f"[Standard VI] {env_name}: {iters_vi} iterations")

    iters_gs, norms_gs = gauss_seidel_vi(env=env, V_star=V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS)
    print(f"[Gauss-Seidel] {env_name}: {iters_gs} iterations")

    iters_ps, norms_ps = prioritized_sweeping_vi(env=env, V_star=V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS)
    print(f"[Prioritized Sweeping] {env_name}: {iters_ps} iterations")
    

    pi_all_norms = []
    pi_all_iters = []
    for i in range(5):
        iters_pi, norms_pi = policy_iteration(env=env, V_star=V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS)
        print(f"[Policy Iteration-{i+1}] {env_name}: {iters_pi} iterations")

        pi_all_iters.append(iters_pi)
        pi_all_norms.append(norms_pi)
        
    avg_pi_iters = np.mean(pi_all_iters)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(norms_std) + 1), norms_std, label=f'Standard VI ({iters_std} iters)', linewidth=2)
    plt.plot(range(1, len(norms_gs) + 1), norms_gs, label=f'Gauss-Seidel ({iters_gs} iters)', linewidth=2)
    plt.plot(range(1, len(norms_ps) + 1), norms_ps, label=f'Prioritized Sweeping ({iters_ps} iters)', linewidth=2)
    
    for i in range(5):
        label = f'Policy Iteration (Avg: {avg_pi_iters:.0f} iters)' if i == 0 else ""
        plt.plot(range(1, len(pi_all_norms[i]) + 1), pi_all_norms[i], 
                 color='purple', alpha=0.4, label=label)
    
    plt.title(f'Convergence Speed: {env_name}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Distance to Optimal Value')
    
    #plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    filename = f"{env_name}_convergence.png"
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print('\n\n')
    generate_plots("Taxi-v3")
    print('\n')
    generate_plots("FrozenLake-v1", is_slippery=True)