import numpy as np
import random
from gridworld import Gridworld

class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=1.0):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = len(env.actions)
        self.q_table_1 = np.zeros((env.rows, env.cols, self.num_actions))
        self.q_table_2 = np.zeros((env.rows, env.cols, self.num_actions))

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy policy based on the sum of both Q-tables.
        """

        r, c = state
        
        # Explore
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploit: Combine both tables to find the true best action
        else:
            combined_q = self.q_table_1[r, c, :] + self.q_table_2[r, c, :]
            return np.argmax(combined_q)

    def learn(self, state, action, reward, next_state):
        """
        Updates one of the two Q-tables randomly
        """

        r, c = state
        next_r, next_c = next_state
        
        if random.random() < 0.5:
            # UPDATE Q-Table 1

            # Use Q1 to find the best next action
            best_next_action = np.argmax(self.q_table_1[next_r, next_c, :])
            # Use Q2 to get the unbiased value of that action
            unbiased_next_q = self.q_table_2[next_r, next_c, best_next_action]
            
            # Update Q1
            current_q = self.q_table_1[r, c, action]
            td_target = reward + self.gamma * unbiased_next_q
            self.q_table_1[r, c, action] += self.alpha * (td_target - current_q)
            
        else:
            # Update Q-Table 2

            # Use Q2 to find the best next action
            best_next_action = np.argmax(self.q_table_2[next_r, next_c, :])
            # Use Q1 to get the unbiased value of that action
            unbiased_next_q = self.q_table_1[next_r, next_c, best_next_action]
            
            # Update Q2
            current_q = self.q_table_2[r, c, action]
            td_target = reward + self.gamma * unbiased_next_q
            self.q_table_2[r, c, action] += self.alpha * (td_target - current_q)

    def train(self, episodes=500):
        """
        Trains the agent over a specified number of episodes.
        """

        steps_per_episode = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done:
                # Choose action using combined Q-tables
                action = self.choose_action(state)
                
                # Step
                next_state, reward, done = self.env.step(action)
                
                # Learn
                self.learn(state, action, reward, next_state)
                
                state = next_state
                steps += 1
                
            steps_per_episode.append(steps)
            
        return steps_per_episode

if __name__ == "__main__":
    env = Gridworld(grid_type='A', use_diagonals=False)
    agent = DoubleQLearningAgent(env=env, alpha=0.1, epsilon=0.1, gamma=1.0)
    episode_steps = agent.train(episodes=500)
    
    print("Steps per episode:", episode_steps)