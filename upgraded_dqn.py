import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from environment import ConnectFourEnv  # Assuming 'envirnment' was a typo
from collections import namedtuple, deque
from tqdm import tqdm
import random

# Deep Q-Network architecture definition using PyTorch
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Define convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Separate streams for calculating value and advantage functions
        self.value_stream = nn.Sequential(
            nn.Linear(32 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(32 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    # Forward pass through the network
    def forward(self, x):
        x = x.view(-1, 1, 6, 7)  # Reshape the input for the convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers

        # Compute value and advantage functions
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combine value and advantages to compute Q-values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# Experience tuple for storing transitions
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay buffer for experience replay mechanism
class ExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha  # Priority level determination
        self.buffer = deque(maxlen=capacity)  # Fixed-size buffer to store experience tuples
        self.priorities = deque(maxlen=capacity)  # Store the priorities for each experience
        self.max_priority = 1.0

    # Function to sample a batch of experiences from the buffer
    def sample(self, batch_size, beta=0.4):
        # Compute probabilities for each experience
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / scaled_priorities.sum()
        sample_indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        samples = [self.buffer[idx] for idx in sample_indices]

        # Compute importance-sampling weights
        weights = (len(self.buffer) * sample_probs[sample_indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return samples, weights, sample_indices

    # Add a new experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(float(self.max_priority))

    # Update priorities of sampled experiences
    def update_priorities(self, indices, errors, offset=0.1):
        for idx, error in zip(indices, errors):
            updated_priority = (error + offset) ** self.alpha
            self.priorities[idx] = float(updated_priority)
            self.max_priority = max(self.max_priority, self.priorities[idx])

# DQN agent for interacting with the environment
class DQNAgent:
    def __init__(self, env, buffer_capacity=100000, batch_size=64, target_update_frequency=10):
        self.env = env  # Environment where the agent operates
        self.model = DQN()  # Online network for evaluating current policy
        self.target_model = DQN()  # Target network for stable Q-value targets
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target network
        self.buffer = ExperienceReplayBuffer(buffer_capacity)  # Buffer for storing experiences
        self.batch_size = batch_size  # Batch size for sampling from the buffer
        self.target_update_frequency = target_update_frequency  # Frequency of target network update
        self.optimizer = optim.Adam(self.model.parameters())  # Optimizer for training the network
        self.loss_fn = nn.MSELoss(reduction='none')  # Loss function with individual loss computation

    # Function to select an action based on current state and policy
    def select_action(self, state, epsilon):
        # Check for available actions in the current state
        available_columns = [col for col in range(7) if state[0][col] == 0]
        if not available_columns:
            return None  # If no actions are available, return None

        # Epsilon-greedy strategy for action selection
        if random.random() < epsilon:
            return random.choice(available_columns)  # Choose a random action

        # Otherwise, choose the best action according to the current policy
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Ensure that the Q-values for unavailable actions are very low
        q_values_clone = q_values.clone()
        q_values_clone[0, [col for col in range(7) if col not in available_columns]] = -float('inf')

        # Choose the action with the highest Q-value
        chosen_action = torch.argmax(q_values_clone).item()
        return chosen_action

    # Training loop for the agent
    def train(self, num_episodes, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=0.9995):
        epsilon = epsilon_start  # Starting value for epsilon in epsilon-greedy strategy

        for episode in tqdm(range(num_episodes)):  # Iterate over each episode
            state = self.env.reset()  # Reset the environment at the start of each episode
            total_reward = 0  # Track the total reward gained in the episode

            for step in range(self.env.max_moves):  # Iterate over each step in the episode
                action = self.select_action(state, epsilon)  # Select an action
                next_state, reward, done, _ = self.env.step(action)  # Take the action in the environment
                self.buffer.add(Experience(state, action, reward, next_state, done))  # Store the experience

                state = next_state  # Move to the next state
                total_reward += reward  # Update the total reward

                # Break the loop if the episode is finished
                if done:
                    break

                # Training step
                if len(self.buffer.buffer) >= self.batch_size:
                    experiences, weights, indices = self.buffer.sample(self.batch_size)
                    # Process the batch of experiences
                    states, actions, rewards, next_states, dones = zip(*experiences)
                    
                    states = torch.stack(states)
                    next_states = torch.stack(next_states)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    dones = torch.tensor(dones, dtype=torch.float32)

                    # Calculate the Q-values and the target Q-values
                    q_values = self.model(states)
                    with torch.no_grad():
                        best_actions = self.model(next_states).argmax(1)
                        next_q_values = self.target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
                        target_q_values = rewards + 0.99 * next_q_values * (1 - dones)

                    # Compute the loss and perform a gradient descent step
                    losses = self.loss_fn(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
                    loss = losses.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update the priorities in the replay buffer
                    self.buffer.update_priorities(indices, losses.detach().numpy())

                # Update the target network periodically
                if step % self.target_update_frequency == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Update epsilon using the decay rate
            epsilon = max(epsilon_final, epsilon * epsilon_decay)
            tqdm.write(f"Episode: {episode}, Total Reward: {total_reward:.4f}, Epsilon: {epsilon:.2f}")

# Entry point of the script
if __name__ == '__main__':
    env = ConnectFourEnv()  # Initialize the Connect Four environment
    dqn_agent = DQNAgent(env)  # Create an instance of the DQN agent

    # Train the agent on a specified number of episodes
    num_episodes = 10000
    dqn_agent.train(num_episodes=num_episodes)

    # Save the trained model parameters
    torch.save({
        'model_state_dict': dqn_agent.model.state_dict(),
        'target_model_state_dict': dqn_agent.target_model.state_dict(),
        'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
    }, 'saved_agents/dqn_agent_after_training.pth')
