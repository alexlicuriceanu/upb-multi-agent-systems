class Gridworld:
    def __init__(self, grid_type='A', use_diagonals=False, num_agents=1):
        self.rows = 7  
        self.cols = 10 
        self.num_agents = num_agents
        
        # Dynamic start states
        if self.num_agents == 1:
            self.start_states = [(3, 0)]
        else:
            self.start_states = [(2, 0), (4, 0)]
            
        self.goal_state = (3, 7)
        self.current_state = None
        
        self.grid_type = grid_type
        self.obstacles = [(1, 5), (2, 5), (3, 5), (4, 5)] 
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] 
        
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        if use_diagonals:
            self.actions.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])

    def reset(self):
        if self.num_agents == 1:
            self.current_state = self.start_states[0]
        else:
            self.current_state = tuple(self.start_states)
        return self.current_state

    def _calculate_next_pos(self, pos, action_idx):
        if pos == self.goal_state: return pos 
        
        row, col = pos
        action = self.actions[action_idx]
        new_row, new_col = row + action[0], col + action[1]
        
        if self.grid_type == 'A':
            if (new_row, new_col) in self.obstacles:
                new_row, new_col = row, col
        elif self.grid_type == 'B':
            wind_strength = self.wind[col]
            new_row -= wind_strength
            
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))
        
        return (new_row, new_col)

    def step(self, actions):
        is_single = (self.num_agents == 1)
        positions = [self.current_state] if is_single else list(self.current_state)
        actions = [actions] if is_single else list(actions)
            
        next_positions = [self._calculate_next_pos(pos, act) for pos, act in zip(positions, actions)]
        
        # Collision resolution for multi-agent
        if not is_single:
            pos1, pos2 = positions
            next_pos1, next_pos2 = next_positions
            
            if next_pos1 == next_pos2 and next_pos1 != self.goal_state:
                next_pos1, next_pos2 = pos1, pos2
            elif next_pos1 == pos2 and next_pos2 == pos1:
                next_pos1, next_pos2 = pos1, pos2
                
            next_positions = [next_pos1, next_pos2]
            
        rewards = []
        dones = []
        
        for i in range(self.num_agents):
            pos = positions[i]
            next_pos = next_positions[i]
            
            done = (next_pos == self.goal_state)
            reward = 1 if done and pos != self.goal_state else (-1 if not done else 0)
            
            rewards.append(reward)
            dones.append(done)
            
        if is_single:
            self.current_state = next_positions[0]
            return self.current_state, rewards[0], dones[0]
        else:
            self.current_state = tuple(next_positions)
            return self.current_state, tuple(rewards), all(dones)

    def render(self):
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        for obs in self.obstacles: grid[obs[0]][obs[1]] = 'X'
        grid[self.goal_state[0]][self.goal_state[1]] = 'G'
        
        if self.num_agents == 1:
            if self.current_state != self.goal_state: grid[self.current_state[0]][self.current_state[1]] = 'A'
        else:
            pos1, pos2 = self.current_state
            if pos1 != self.goal_state: grid[pos1[0]][pos1[1]] = '1'
            if pos2 != self.goal_state: grid[pos2[0]][pos2[1]] = '2'
            
        for row in grid: print(' '.join(row))
        print()