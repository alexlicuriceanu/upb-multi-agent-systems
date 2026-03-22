import numpy as np
import gym
from standard import standard_vi
from constants import GAMMA, EPSILON, MAX_ITERATIONS

def gauss_seidel_vi(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    # extract number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n

    # initialize the value function to zeros
    V = np.zeros(nS)
    iteration_count = 0

    # keep track of the norm of the difference between V and V_star at each iteration
    norms_history = []
    current_norm = np.linalg.norm(V - V_star)
    
    # start the iteration loop
    while iteration_count < max_iters:
        delta = 0 

        # loop through each state (s in S) 
        for s in range(nS):
            action_values = np.zeros(nA)

            # loop through each possible action to find the best one
            for a in range(nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    # calculate the expected value for each action using the transition probabilities
                    # using the most up-to-date values of V
                    action_values[a] += prob * (reward + gamma * V[next_state])
            
            # update V(s) with the maximum action value
            old_v = V[s]
            V[s] = np.max(action_values)

            # calculate the maximum change in value for the stopping criterion
            delta = max(delta, abs(V[s] - old_v))
            iteration_count += 1
            
            # keep track of the norm of the difference between V and V_star at each iteration
            current_norm = np.linalg.norm(V - V_star)
            norms_history.append(current_norm)
            
            if iteration_count >= max_iters:
                break

        # check if the algorithm has converged
        if delta <= epsilon or current_norm <= epsilon:
            break

    return iteration_count, norms_history

if __name__ == "__main__":
    env_taxi = gym.make("Taxi-v3")
    V_star_taxi, iters_taxi = standard_vi(env_taxi)
    iters_gs, norms_gs = gauss_seidel_vi(env_taxi, V_star_taxi)
    print(f"[Gauss-Seidel VI] Taxi-v3: {iters_gs} iterations")

    env_frozen = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    V_star_frozen, iters_frozen = standard_vi(env_frozen)
    iters_gs_frozen, norms_gs_frozen = gauss_seidel_vi(env_frozen, V_star_frozen)
    print(f"[Gauss-Seidel VI] FrozenLake-v1: {iters_gs_frozen} iterations")
    