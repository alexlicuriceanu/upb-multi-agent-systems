import numpy as np
import random
from gridworld import Gridworld

class QLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=1.0):
        """
        Initializes the Q-Learning agent with environment parameters and learning rates.
        """

        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = len(env.actions)
        self.q_table = np.zeros((env.rows, env.cols, self.num_actions))

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        """

        row, col = state
        
        # explore: choose a random action
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # exploit: choose the action with the highest Q-value for the current state
        else:
            return np.argmax(self.q_table[row, col])

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning update rule.
        Gamma is 1 as it is an undiscounted task.
        """

        r, c = state
        next_r, next_c = next_state
        
        # get the current Q-value
        current_q = self.q_table[r, c, action]
        
        # get the maximum possible Q-value for the next state
        max_next_q = np.max(self.q_table[next_r, next_c])
        
        # calculate the new Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # update the table
        self.q_table[r, c, action] = new_q

    def train(self, episodes=500):
        """
        Trains the agent over a specified number of episodes.
        Returns data useful for Task 1's performance metrics[cite: 20].
        """
        steps_per_episode = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done:
                # choose an action
                action = self.choose_action(state)
                
                # take a step in the environment
                next_state, reward, done = self.env.step(action)
                
                # learn from the experience
                self.learn(state, action, reward, next_state)
                
                # transition to the next state
                state = next_state
                steps += 1
                
            steps_per_episode.append(steps)
            
        return steps_per_episode

if __name__ == "__main__":
    env = Gridworld(grid_type='A', use_diagonals=False)
    agent = QLearningAgent(env=env, alpha=0.1, epsilon=0.1, gamma=1.0)
    episode_steps = agent.train(episodes=500)
    
    print("Steps per episode:", episode_steps)