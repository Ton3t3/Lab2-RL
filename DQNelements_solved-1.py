# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 3 at KTH, Stockholm.

import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter


"""Tensorboard instructions:
    1. Install library using conda install tensorboard
    2. After running the script, open the command line (in anaconda) 
    3. Run tensorboard --logdir=runs/cartpole_dqn/
    4. Copy the address that you get - probably http://localhost:6006/
    5. You should be able to see logged losses, rewards and epsilon values. 
    6. If you want to log in additional quantities use writer.add_scalar() as below"""
    
writer = SummaryWriter(log_dir="runs/cartpole_dqn")


# Define Experience tuple
# Experience represents a transition in the environment, including the current state, action taken,
# received reward, next state, and whether the episode is done.
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:
    """Replay buffer for storing experiences.
    
       The experience replay buffer stores past experiences so that the agent can learn from them later.
       By sampling randomly from these experiences, the agent avoids overfitting to the most recent 
       transitions and helps stabilize training.
       - The buffer size is limited, and older experiences are discarded to make room for new ones.
       - Experiences are stored as tuples of (state, action, reward, next_state, done).
       - A batch of experiences is sampled randomly during each training step for updating the Q-values."""

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)  # Using deque ensures efficient removal of oldest elements

    def append(self, experience):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)

    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

    def sample_batch(self, n):
        """Randomly sample a batch of experiences"""
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        indices = np.random.choice(len(self.buffer), size=n, replace=False)  # Random sampling
        batch = [self.buffer[i] for i in indices]  # Create a batch from sampled indices
        return zip(*batch)  # Unzip batch into state, action, reward, next_state, and done


class MyNetwork(nn.Module):
    """Feedforward neural network that approximates the Q-function.
    
       The network takes the current state as input and outputs Q-values for all possible actions.
       The action corresponding to the highest Q-value is considered the optimal action.
       - The input size corresponds to the state dimension of the environment.
       - The network has one hidden layer with 64 neurons and ReLU activation.
       - The output layer has one neuron per action (Q-values for each action)."""
        
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)  # First layer: state -> hidden layer
        self.hidden_layer = nn.Linear(64, 64)  # Second layer: hidden -> hidden layer
        self.output_layer = nn.Linear(64, output_size)  # Output layer: hidden -> Q-values
        self.activation = nn.ReLU()  # ReLU activation function for hidden layers

    def forward(self, x):
        """Define forward pass"""
        x = self.activation(self.input_layer(x))  # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x))  # Apply hidden layer and ReLU
        return self.output_layer(x)  # Return Q-values for all actions


### Parameters ###
GAMMA = 0.99  # Discount factor (how much future rewards are considered)
EPSILON = 1.0  # Initial exploration rate (balance between exploration and exploitation)
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for epsilon after each episode
BATCH_SIZE = 32  # Number of experiences to sample from the replay buffer per update
BUFFER_SIZE = 10000  # Size of the replay buffer
LEARNING_RATE = 0.001  # Learning rate for the optimizer
N_EPISODES = 250  # Number of training episodes
MAX_STEPS = 200  # Maximum number of steps per episode

# Initialize environment, buffer, network, and optimizer
env = gym.make('CartPole-v1')  # Create the CartPole environment

# Initialize experience replay buffer
buffer = ExperienceReplayBuffer(maximum_length=BUFFER_SIZE)

# Initialize the Q-network (state -> Q-values for actions)
network = MyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)

# Optimizer for training the Q-network
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)  # Adam optimizer for efficient training



def select_action(state, epsilon):
    """Epsilon-greedy action selection
    # We balance exploration and exploitation using epsilon-greedy.
    # Exploration: Choose a random action.
    # Exploitation: Choose the action with the highest Q-value (the optimal action)."""
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore by selecting a random action
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32)  # Convert state to tensor
        return network(state_tensor).argmax().item()  # Exploit by selecting the action with max Q-value

# Training loop
for episode in range(N_EPISODES):
    state = env.reset()[0]  # Reset environment and get initial state
    total_reward = 0
    for t in range(MAX_STEPS):
        # Choose action using epsilon-greedy policy
        action = select_action(state, EPSILON)

        # Execute action in environment and get feedback (next state, reward, etc.)
        next_state, reward, terminal, truncated, _ = env.step(action)
        done = terminal or truncated  # Done is True if episode ends
        total_reward += reward

        # Store the experience (state, action, reward, next state, done) in the buffer
        buffer.append(Experience(state, action, reward, next_state, done))
        state = next_state  # Update state for the next step

        # Training step: update Q-values using a batch of experiences from the buffer
        if len(buffer) >= BATCH_SIZE:
            # Sample a batch of experiences from the buffer
            states, actions, rewards, next_states, dones = buffer.sample_batch(BATCH_SIZE)

            # Convert the batch data into tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # Unsqueeze for correct shape
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q-values for the current states
            q_values = network(states).gather(1, actions).squeeze()  # Q-values for taken actions

            # Compute the target Q-values for the next states
            with torch.no_grad():  # No need to compute gradients for target Q-values
                next_q_values = network(next_states).max(1)[0]  # Max Q-value for next state
                targets = rewards + GAMMA * next_q_values * (1 - dones)  # Target: Bellman equation

            # Compute the loss (MSE loss between predicted Q-values and target Q-values)
            loss = nn.functional.mse_loss(q_values, targets)
               
            # Backpropagation step: update network parameters
            optimizer.zero_grad()  # Zero gradients before backpropagation
            loss.backward()  # Compute gradients
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradients
            optimizer.step()  # Update parameters

            # Log loss to TensorBoard for visualization
            writer.add_scalar("Loss", loss.item(), episode * MAX_STEPS + t)

        if done:  # If the episode ends
            break

    # Decay epsilon: reduce exploration over time
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    
    # Log total reward and epsilon to TensorBoard
    writer.add_scalar("Total Reward", total_reward, episode)
    writer.add_scalar("Epsilon", EPSILON, episode)

    # Print progress for each episode
    print(f"Episode {episode + 1}/{N_EPISODES}: Total Reward: {total_reward}")
    if total_reward >= 200:  # If the agent achieves good performance, stop early
        break

# Close the environment after training
env.close()


# Evaluate the trained policy by rendering it
env = gym.make('CartPole-v1', render_mode="human")
state = env.reset()[0]
done = False
total_reward = 0

# Run the trained agent in the environment
while not done:
    env.render()
    action = select_action(state, 0)  # Choose action (epsilon=0, i.e., exploit the policy)
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    total_reward += reward

print(f"Evaluation Total Reward: {total_reward}")
env.close()

writer.close()

