import gym
import numpy as np

MAX_ITERATIONS = 5e5
EPSILON = 1e-3
GAMMA = 0.9

def standard_value_iteration(env, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    # extract number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # initialize the value function to zeros
    V = np.zeros(nS)
    iteration_count = 0
    
    # standard value iteration algorithm loop
    while iteration_count < max_iters:
        delta = 0
        V_old = np.copy(V) # save the old values for this sweep
        
        # loop through each state (s in S)
        for s in range(nS):
            action_values = np.zeros(nA)
            
            # loop through each possible action to find the best one
            for a in range(nA):
                # calculate the expected value for each action using the transition probabilities
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V_old[next_state])
            
            # update V(s) with the maximum action value
            best_action_value = np.max(action_values)
            
            # calculate the maximum change in value for the stopping criterion
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
            
            # every update to an individual state counts as one iteration
            iteration_count += 1
            
            if iteration_count >= max_iters:
                break
                
        # check if the algorithm has converged
        if delta <= epsilon:
            break
            
    return V, iteration_count

if __name__ == "__main__":
    print("Running Value Iteration for Taxi-v3")
    env_taxi = gym.make("Taxi-v3")
    V_star_taxi, iters_taxi = standard_value_iteration(env_taxi)
    print(f"Taxi-v3 converged in {iters_taxi} iterations\n")
    
    print("Running Value Iteration for FrozenLake-v1 (8x8)")
    env_frozen = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    V_star_frozen, iters_frozen = standard_value_iteration(env_frozen)
    print(f"FrozenLake-v1 converged in {iters_frozen} iterations\n")