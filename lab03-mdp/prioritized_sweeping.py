import numpy as np
import gym
from constants import GAMMA, EPSILON, MAX_ITERATIONS
from standard import standard_vi

def prioritized_sweeping_vi(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    # extract number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n

    # initialize the value function to zeros and the priority queue
    V = np.zeros(nS)
    H = np.zeros(nS)

    # keep track of the norm of the difference between V and V_star at each iteration
    iteration_count = 0
    norms_history = []
    
    def compute_bellman_error(state, current_V):
        # compute the Bellman error for a given state using the current value function
        action_values = np.zeros(nA)

        for a in range(nA):
            for prob, next_state, reward, _ in env.P[state][a]:
                action_values[a] += prob * (reward + gamma * current_V[next_state])

        return np.abs(np.max(action_values) - current_V[state])

    # precompute predecessor map; predecessors[s] = states that can transition into s
    predecessors = {s: set() for s in range(nS)}
    for s in range(nS):
        for a in range(nA):
            for prob, next_state, reward, _ in env.P[s][a]:
                if prob > 0:
                    predecessors[next_state].add(s)

    # initialize the priority queue with the Bellman error of all states
    for s in range(nS):
        H[s] = compute_bellman_error(s, V)
        
    # start the iteration loop
    while iteration_count < max_iters:
        s_k = np.argmax(H)  # state with the highest priority aka largest Bellman error
        delta = H[s_k]  # the Bellman error of the state being updated

        action_values = np.zeros(nA)

        # loop through each possible action from s_k
        for a in range(nA):
            for prob, next_state, reward, _ in env.P[s_k][a]:
                # calculate the expected value for each action using the transition probabilities
                action_values[a] += prob * (reward + gamma * V[next_state])

        # update V(s_k) with the maximum action value
        V[s_k] = np.max(action_values)

        # reset the priority of the updated state
        H[s_k] = 0

        iteration_count += 1
        
        # only update priorities for states that can reach s_k
        for pred in predecessors[s_k]:
            H[pred] = compute_bellman_error(pred, V)
            
        # keep track of the norm of the difference between V and V_star at each iteration
        current_norm = np.linalg.norm(V - V_star)
        norms_history.append(current_norm)
        
        # check if the algorithm has converged
        if delta <= epsilon or current_norm <= epsilon:
            break
        
    return iteration_count, norms_history

if __name__ == "__main__":
    env_taxi = gym.make("Taxi-v3")
    V_star_taxi, iters_taxi = standard_vi(env_taxi)
    iters_ps, norms_ps = prioritized_sweeping_vi(env_taxi, V_star_taxi)
    print(f"[Prioritized Sweeping VI] Taxi-v3: {iters_ps} iterations\n")
    
    env_frozen = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    V_star_frozen, iters_frozen = standard_vi(env_frozen)
    iters_ps_frozen, norms_ps_frozen = prioritized_sweeping_vi(env_frozen, V_star_frozen)
    print(f"[Prioritized Sweeping VI] FrozenLake-v1: {iters_ps_frozen} iterations\n")
