import numpy as np
import random

class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=1.0, num_agents=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_agents = num_agents
        self.num_actions = len(env.actions)
        
        # Dynamically create the shape: (7, 10, 4) for 1 agent, (7, 10, 7, 10, 4) for 2 agents
        shape = (env.rows, env.cols) * self.num_agents + (self.num_actions,)
        self.q_table_1 = np.zeros(shape)
        self.q_table_2 = np.zeros(shape)

    def _get_state_idx(self, state, agent_idx=0):
        if self.num_agents == 1:
            return state  # Returns (row, col)
        
        # For 2 agents, orient so the current agent views itself first
        my_pos = state[agent_idx]
        other_pos = state[1 - agent_idx]
        return my_pos + other_pos  # Flattens to (my_row, my_col, other_row, other_col)

    def choose_action(self, state, agent_idx=0):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
            
        state_idx = self._get_state_idx(state, agent_idx)
        
        # Combine both tables to find the best action
        combined_q = self.q_table_1[state_idx] + self.q_table_2[state_idx]
        return np.argmax(combined_q)

    def learn(self, state, action, reward, next_state, next_action=None, agent_idx=0):
        # Parse the states into clean tuples
        state_idx = self._get_state_idx(state, agent_idx)
        next_state_idx = self._get_state_idx(next_state, agent_idx)
        
        # Append the action to pinpoint the exact current Q-value index
        action_idx = state_idx + (action,)
        
        # Perform Double Q-Learning math
        if random.random() < 0.5:
            # Update Q1
            best_next = np.argmax(self.q_table_1[next_state_idx])
            next_action_idx = next_state_idx + (best_next,)
            
            unbiased_q = self.q_table_2[next_action_idx]
            cur_q = self.q_table_1[action_idx]
            
            self.q_table_1[action_idx] += self.alpha * ((reward + self.gamma * unbiased_q) - cur_q)
        else:
            # Update Q2
            best_next = np.argmax(self.q_table_2[next_state_idx])
            next_action_idx = next_state_idx + (best_next,)
            
            unbiased_q = self.q_table_1[next_action_idx]
            cur_q = self.q_table_2[action_idx]
            
            self.q_table_2[action_idx] += self.alpha * ((reward + self.gamma * unbiased_q) - cur_q)