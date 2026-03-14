import numpy as np
import gym
from constants import GAMMA, EPSILON, MAX_ITERATIONS
from standard import standard_value_iteration

def prioritized_sweeping_vi(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    H = np.zeros(nS)
    iteration_count = 0
    norms_history = []
    
    def compute_bellman_error(state, current_V):
        action_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                action_values[a] += prob * (reward + gamma * current_V[next_state])
        return np.abs(np.max(action_values) - current_V[state])
        
    for s in range(nS):
        H[s] = compute_bellman_error(s, V)
        
    while iteration_count < max_iters:
        s_k = np.argmax(H)
        action_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[s_k][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        V[s_k] = np.max(action_values)
        iteration_count += 1
        
        for s in range(nS):
            H[s] = compute_bellman_error(s, V)
            
        current_norm = np.linalg.norm(V - V_star)
        norms_history.append(current_norm)
        
        if current_norm <= epsilon:
            break
    return iteration_count, norms_history

# --- EXECUTION SCRIPT ---
if __name__ == "__main__":
    # We will test on Taxi-v3 first
    print("Initializing Taxi-v3...")
    env_taxi = gym.make("Taxi-v3")
    
    print("1. Running Standard Value Iteration to find V*...")
    V_star_taxi, iters_taxi = standard_value_iteration(env_taxi)
    
    print("3. Running Prioritized Sweeping VI... (This may take a moment)")
    iters_ps, norms_ps = prioritized_sweeping_vi(env_taxi, V_star_taxi)
    print(f"   -> Prioritized Sweeping converged in {iters_ps} iterations.\n")
    
    print("Done! You now have the iteration counts for your report.")
