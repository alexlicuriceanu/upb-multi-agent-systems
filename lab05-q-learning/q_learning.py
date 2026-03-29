import numpy as np
import gymnasium as gym

def evaluate_policy(env, q_table, eval_epochs=50):
    total_rewards = []
    
    for _ in range(eval_epochs):
        state, _ = env.reset()
        done = False
        epoch_reward = 0
        
        while not done:
            # always choose the action with the highest Q-value
            action = np.argmax(q_table[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            epoch_reward += reward
            done = terminated or truncated
            state = next_state
            
        total_rewards.append(epoch_reward)
        
    return np.mean(total_rewards)

def train_q_learning(env_name, alpha, gamma, epsilon, epochs, eval_freq=100, eval_epochs=50):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    
    train_rewards = []
    eval_metrics = {'epochs': [], 'avg_rewards': []}
    
    for epoch in range(1, epochs + 1):
        state, _ = env.reset()
        done = False
        total_train_reward = 0
        
        while not done:
            # epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # explore
            else:
                action = np.argmax(q_table[state, :]) # exploit
                
            # take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update rule
            best_next_action = np.argmax(q_table[next_state, :])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (not done)
            td_error = td_target - q_table[state, action]
            
            q_table[state, action] += alpha * td_error
            
            total_train_reward += reward
            state = next_state
            
        train_rewards.append(total_train_reward)
        
        # periodic evaluation
        if epoch % eval_freq == 0 or epoch == epochs:
            avg_eval_reward = evaluate_policy(eval_env, q_table, eval_epochs)
            eval_metrics['epochs'].append(epoch)
            eval_metrics['avg_rewards'].append(avg_eval_reward)
            
    env.close()
    eval_env.close()
    
    return q_table, train_rewards, eval_metrics

if __name__ == "__main__":
    q_table, train_rwds, evals = train_q_learning(
        env_name="Taxi-v3",
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        epochs=5000,
        eval_freq=100
    )
    print(f"[Taxi-v3] Final Eval Reward: {evals['avg_rewards'][-1]}")