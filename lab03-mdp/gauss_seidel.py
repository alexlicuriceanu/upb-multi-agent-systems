import numpy as np
import gym
from standard import standard_value_iteration
from constants import GAMMA, EPSILON, MAX_ITERATIONS


def gauss_seidel_vi(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    iteration_count = 0
    norms_history = []
    current_norm = np.linalg.norm(V - V_star)
    
    while iteration_count < max_iters:
        delta = 0  # max per-state change this sweep
        for s in range(nS):
            action_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            
            old_v = V[s]
            V[s] = np.max(action_values)
            delta = max(delta, abs(V[s] - old_v))
            iteration_count += 1
            
            current_norm = np.linalg.norm(V - V_star)
            norms_history.append(current_norm)
            
            if iteration_count >= max_iters:
                break

        # Stop if either the per-state delta is tiny (like Standard VI)
        # or if we've actually gotten close to V_star in norm
        if delta <= epsilon or current_norm <= epsilon:
            break
    return iteration_count, norms_history

if __name__ == "__main__":
    env_taxi = gym.make("Taxi-v3")
    
    # FIX: Force a super-strict epsilon (e.g., 1e-8) to get a hyper-precise V*
    print("Generating precise V* for Taxi-v3...")
    V_star_taxi, iters_taxi = standard_value_iteration(env_taxi, epsilon=1e-8)
    
    # Run Gauss-Seidel using the normal epsilon (1e-3) from your constants
    print("Running Gauss-Seidel VI...")
    iters_gs, norms_gs = gauss_seidel_vi(env_taxi, V_star_taxi)
    print(f"[Gauss-Seidel VI] Taxi-v3: {iters_gs} iterations")

    # Repeat the same fix for FrozenLake
    env_frozen = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    print("Generating precise V* for FrozenLake-v1...")
    V_star_frozen, iters_frozen = standard_value_iteration(env_frozen, epsilon=1e-8)
    
    iters_gs_frozen, norms_gs_frozen = gauss_seidel_vi(env_frozen, V_star_frozen)
    print(f"[Gauss-Seidel VI] FrozenLake-v1: {iters_gs_frozen} iterations")
    