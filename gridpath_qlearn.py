import numpy as np
import matplotlib.pyplot as plt

# Define grid dimensions
grid_size = 5
goal_position = (grid_size - 1, grid_size - 1)
trap_positions = [(1, 2), (2, 3), (3, 1)]  # Example trap positions

# Q-Table initialization
q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 2000

# Action definitions
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Function to take an action in the grid
def take_action(state, action):
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    # Ensure the agent stays within grid bounds
    new_state = (max(0, min(grid_size - 1, new_state[0])), max(0, min(grid_size - 1, new_state[1])))
    return new_state

# Training loop
for episode in range(num_episodes):
    state = (0, 0)  # Start position
    done = False
    
    while not done:
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(4)  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit

        # Take action and observe the next state and reward
        new_state = take_action(state, action)
        
        # Set rewards
        if new_state == goal_position:
            reward = 10  # Reward for reaching the goal
            done = True
        elif new_state in trap_positions:
            reward = -10  # Penalty for hitting a trap
        else:
            reward = -1  # Small penalty for each move to encourage shortest path

        # Q-Learning update
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[new_state[0], new_state[1]])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state[0], state[1], action] = new_value

        state = new_state  # Move to the next state

# Display learned Q-values
print("Training completed!")

# Testing the learned policy
state = (0, 0)  # Start position
path = [state]
done = False

while not done:
    action = np.argmax(q_table[state[0], state[1]])
    state = take_action(state, action)
    path.append(state)
    
    if state == goal_position:
        done = True

print("Path taken by the agent:", path)

# Visualize path
grid = np.zeros((grid_size, grid_size))
for trap in trap_positions:
    grid[trap] = -1  # Mark traps
grid[goal_position] = 1  # Mark goal
for step in path:
    grid[step] = 0.5  # Mark path

plt.imshow(grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Grid Cell Value")
plt.title("Agent Path in Gridworld")
plt.show()
