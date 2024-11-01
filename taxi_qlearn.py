import numpy as np
import gym
import random
from IPython.display import clear_output
import time

# Load the Taxi environment from OpenAI Gym
env = gym.make("Taxi-v3", render_mode="human")

# Q-Table initialization
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

# Training
num_episodes = 250

for episode in range(num_episodes):
    # Use only the state ID if reset returns a tuple
    state = env.reset() if isinstance(env.reset(), int) else env.reset()[0]
    done = False
    
    while not done:
        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        
        # Take action and observe result
        result = env.step(action)
        next_state = result[0] if isinstance(result, tuple) else result  # Next state ID
        reward = result[1]
        done = result[2]
        
        # Update Q-Table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        # Move to the next state
        state = next_state

print("Training completed!")

# Testing the agent
state = env.reset() if isinstance(env.reset(), int) else env.reset()[0]
env.render()
done = False

for step in range(20):
    clear_output(wait=True)
    print("Step:", step + 1)
    action = np.argmax(q_table[state])
    result = env.step(action)
    state = result[0] if isinstance(result, tuple) else result
    reward = result[1]
    done = result[2]
    
    env.render()
    time.sleep(1)
    
    if done:
        break

env.close()
