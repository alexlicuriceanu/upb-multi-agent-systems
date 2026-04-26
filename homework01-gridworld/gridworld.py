class Gridworld:
    def __init__(self, grid_type='A', use_diagonals=False):
        # dimensions
        self.rows = 7  
        self.cols = 10 
        
        # start and goal states
        self.start_state = (3, 0)
        self.goal_state = (3, 7)
        self.current_state = self.start_state
        
        self.grid_type = grid_type
        
        # obstacles
        self.obstacles = [(1, 5), (2, 5), (3, 5), (4, 5)] 
        
        # wind
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] 
        
        # actions; 0: up, 1: right; 2: down; 3: left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # add diagonal actions for task 2 
        if use_diagonals:
            # 4: up, right; 5: down, right; 6: down, left; 7: up, left
            self.actions.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])

    def reset(self):
        """Resets the agent to the starting position at the beginning of an episode."""

        self.current_state = self.start_state
        return self.current_state

    def step(self, action_idx):
        """Applies the action, calculates the wind/obstacles, and returns the new state and reward."""

        row, col = self.current_state
        action = self.actions[action_idx]
        
        # calculate new position
        new_row = row + action[0]
        new_col = col + action[1]
        
        if self.grid_type == 'A':
            # check if we hit an obstacle
            if (new_row, new_col) in self.obstacles:
                new_row, new_col = row, col
                
        elif self.grid_type == 'B':
            # shift the agent up by wind strength positions
            wind_strength = self.wind[col]
            new_row -= wind_strength
            
        # handle boundaries
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))
        
        self.current_state = (new_row, new_col)
        
        if self.current_state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = -1
            done = False
            
        return self.current_state, reward, done

    def render(self):
        """Prints the current state of the gridworld."""

        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # mark obstacles
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        # mark start and goal states
        grid[self.start_state[0]][self.start_state[1]] = 'S'
        grid[self.goal_state[0]][self.goal_state[1]] = 'G'
        
        # mark agent position
        if self.current_state != self.start_state and self.current_state != self.goal_state:
            grid[self.current_state[0]][self.current_state[1]] = 'A'
        
        for row in grid:
            print(' '.join(row))

        print()

if __name__ == "__main__":
    env = Gridworld(grid_type='A', use_diagonals=True)
    state = env.reset()
    print(f"Starting State: {state}")
    
    actions = [1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1]
    for action in actions:
        env.render()

        new_state, reward, done = env.step(action)
        print(f"Action: {action}, New State: {new_state}, Reward: {reward}, Done: {done}")

        if done:
            break

    env.render()