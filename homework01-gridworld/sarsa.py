import numpy as np
import random
from gridworld import Gridworld

class SarsaAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=1.0):
        """
        Initializes the SARSA agent.
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

    def learn(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-table using the SARSA update rule.
        """

        r, c = state
        next_r, next_c = next_state
        
        # get the current Q-value
        current_q = self.q_table[r, c, action]
        
        # get the Q-value for the actual next action chosen by the policy
        next_q = self.q_table[next_r, next_c, next_action]
        
        # calculate the new Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        # update the table
        self.q_table[r, c, action] = new_q

    def train(self, episodes=500):
        """
        Trains the agent over a specified number of episodes.
        """

        steps_per_episode = []
        
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            
            done = False
            steps = 0
            
            while not done:
                # take a step in the environment
                next_state, reward, done = self.env.step(action)
                
                # choose the next action based on the new state
                next_action = self.choose_action(next_state)
                
                # learn from the experience
                self.learn(state, action, reward, next_state, next_action)
                
                # transition both the state and the action
                state = next_state
                action = next_action
                
                steps += 1
                
            steps_per_episode.append(steps)
            
        return steps_per_episode

if __name__ == "__main__":   
    env = Gridworld(grid_type='A', use_diagonals=False)
    agent = SarsaAgent(env=env, alpha=0.1, epsilon=0.1, gamma=1.0)
    episode_steps = agent.train(episodes=500)
    
    print("Steps per episode:", episode_steps)