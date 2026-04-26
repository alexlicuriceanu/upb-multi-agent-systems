class Gridworld:
    def __init__(self, grid_type='A', use_diagonals=False):
        # Define Grid Dimensions
        self.rows = 7  
        self.cols = 10 
        
        # Define Start and Goal States
        self.start_state = (3, 0)
        self.goal_state = (3, 7)
        self.current_state = self.start_state
        
        self.grid_type = grid_type
        
        # Define Gridworld A Obstacles
        self.obstacles = [(1, 5), (2, 5), (3, 5), (4, 5)] 
        
        # Define Gridworld B Wind
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] 
        
        # Define Actions
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Add diagonal actions for task 2 
        if use_diagonals:
            # Up-Right, 5: Down-Right, 6: Down-Left, 7: Up-Left
            self.actions.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])

    def reset(self):
        """Resets the agent to the starting position at the beginning of an episode."""

        self.current_state = self.start_state
        return self.current_state

    def step(self, action_idx):
        """Applies the action, calculates the wind/obstacles, and returns the new state and reward."""

        row, col = self.current_state
        action = self.actions[action_idx]
        
        # Calculate intended new position
        new_row = row + action[0]
        new_col = col + action[1]
        
        # Apply Gridworld mechanics
        if self.grid_type == 'A':
            # Check if the intended move hits an obstacle
            if (new_row, new_col) in self.obstacles:
                new_row, new_col = row, col # Stay in place if blocked
                
        elif self.grid_type == 'B':
            # Apply wind shifting the state upward
            wind_strength = self.wind[col]
            new_row -= wind_strength
            
        # Handle grid boundaries
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))
        
        self.current_state = (new_row, new_col)
        
        # Determine Reward and Done Flag
        if self.current_state == self.goal_state:
            reward = 1   # Reward for reaching the goal
            done = True  # Episode is over
        else:
            reward = -1  # Constant step penalty
            done = False
            
        return self.current_state, reward, done

    def render(self):
        """Prints the current state of the gridworld."""
        
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        # Mark start and goal states
        grid[self.start_state[0]][self.start_state[1]] = 'S'
        grid[self.goal_state[0]][self.goal_state[1]] = 'G'
        
        # Mark current state
        if self.current_state != self.start_state and self.current_state != self.goal_state:
            grid[self.current_state[0]][self.current_state[1]] = 'A'
        
        # Print the grid
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