import numpy as np

# Parameters
stock_max = 100  # Maximum stock level
order_max = 20  # Maximum order per step
demand_prob = 0.1  # Probability of demand for each unit of stock

# Initialize Q-Table
q_table = np.zeros((stock_max + 1, order_max + 1))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Training
num_episodes = 100
for episode in range(num_episodes):
    stock = stock_max // 2  # Start with half-stock
    done = False
    
    while not done:
        # Choose action (order amount)
        if np.random.rand() < epsilon:
            order = np.random.randint(0, order_max + 1)  # Explore
        else:
            order = np.argmax(q_table[stock])  # Exploit
        
        # Receive demand
        demand = np.random.binomial(stock, demand_prob)
        
        # Calculate next stock level
        new_stock = max(0, min(stock_max, stock + order - demand))
        
        # Reward function (penalty for overstock and shortage)
        reward = - (0.5 * (stock - demand) ** 2 + 0.1 * order ** 2)
        
        # Q-learning update
        old_value = q_table[stock, order]
        next_max = np.max(q_table[new_stock])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[stock, order] = new_value
        
        # Move to the next stock level
        stock = new_stock
        
        # Termination condition
        done = episode >= num_episodes

print("Training completed!")

# Testing the policy
stock = stock_max // 2
for step in range(10):
    order = np.argmax(q_table[stock])
    demand = np.random.binomial(stock, demand_prob)
    stock = max(0, min(stock_max, stock + order - demand))
    print(f"Step {step+1}: Stock level = {stock}, Order = {order}, Demand = {demand}")
