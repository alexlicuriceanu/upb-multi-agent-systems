import numpy as np
import gym
from constants import GAMMA, EPSILON, MAX_ITERATIONS
from standard import standard_vi

def policy_iteration(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iters=MAX_ITERATIONS):
    # extract number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # initialize a random policy and the value function to zeros
    policy = np.random.choice(nA, nS)
    V = np.zeros(nS)
    
    # keep track of the norm of the difference between V and V_star at each iteration
    iteration_count = 0
    norms_history = []
    policy_stable = False
    
    # start the iteration loop
    while not policy_stable and iteration_count < max_iters:
        while True:
            delta = 0

            for s in range(nS):
                v = V[s]
                a = policy[s]
                
                # compute new value based on the current policy
                new_v = 0
                for prob, next_state, reward, _ in env.P[s][a]:
                    new_v += prob * (reward + gamma * V[next_state])
                    
                V[s] = new_v
                
                # every update to an individual state counts as one iteration 
                iteration_count += 1
                
                # keep track of the norm of the difference between V and V_star at each iteration
                current_norm = np.linalg.norm(V - V_star)
                norms_history.append(current_norm)
                
                # calculate the maximum change in value for the stopping criterion
                delta = max(delta, np.abs(v - V[s]))
                
                if iteration_count >= max_iters:
                    break
            
            if delta <= epsilon or iteration_count >= max_iters:
                break
                
        if iteration_count >= max_iters:
            break
            
        # policy improvement
        policy_stable = True

        for s in range(nS):
            old_action = policy[s]
            action_values = np.zeros(nA)
            
            # evaluate all possible actions to find the best one
            for a in range(nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
                    
            # update the policy to the best action
            best_action = np.argmax(action_values)
            policy[s] = best_action
            
            # if the action changed, then the policy is not stable yet
            if old_action != best_action:
                policy_stable = False
                
    return iteration_count, norms_history

if __name__ == "__main__":
    
    env_taxi = gym.make("Taxi-v3")
    V_star_taxi, _ = standard_vi(env_taxi, epsilon=1e-3)
    
    taxi_iters = []
    for i in range(5):
        iters, _ = policy_iteration(env_taxi, V_star_taxi)
        taxi_iters.append(iters)
        print(f"Run {i+1}: Converged in {iters} iterations.")
        
    print(f"[Policy Iteration] Taxi-v3 average: {np.mean(taxi_iters)} iterations\n")
    

    env_frozen = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    V_star_frozen, _ = standard_vi(env_frozen, epsilon=1e-3)

    frozen_iters = []
    for i in range(5):
        iters, _ = policy_iteration(env_frozen, V_star_frozen)
        frozen_iters.append(iters)
        print(f"Run {i+1}: Converged in {iters} iterations.")
        
    print(f"[Policy Iteration] FrozenLake-v1 average: {np.mean(frozen_iters)} iterations\n")