import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=1.0, num_agents=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_agents = num_agents
        self.num_actions = len(env.actions)
        
        # Dynamically create the shape: (7, 10, 4) for 1 agent, (7, 10, 7, 10, 4) for 2 agents
        shape = (env.rows, env.cols) * self.num_agents + (self.num_actions,)
        self.q_table = np.zeros(shape)

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
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, next_action=None, agent_idx=0):
        # Parse the states into clean tuples
        state_idx = self._get_state_idx(state, agent_idx)
        next_state_idx = self._get_state_idx(next_state, agent_idx)
        
        # Append the action to the state tuple to pinpoint the exact Q-value
        action_idx = state_idx + (action,)
        
        # Perform standard Q-Learning math
        current_q = self.q_table[action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        
        self.q_table[action_idx] += self.alpha * (reward + self.gamma * max_next_q - current_q)