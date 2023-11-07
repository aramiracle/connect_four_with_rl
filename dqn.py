import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from environment import ConnectFourEnv

# Define the Deep Q-Network architecture
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)    # Second fully connected layer
        self.fc3 = nn.Linear(128, 7)      # Output layer for action values

    def forward(self, x):
        x = x.view(-1, 6 * 7)             # Flatten the input for the fully connected layer
        x = torch.relu(self.fc1(x))       # Activation function after first layer
        x = torch.relu(self.fc2(x))       # Activation function after second layer
        return self.fc3(x)                # Return the output of the network

# Implement the experience replay buffer for more efficient learning
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Define a double-ended queue with fixed capacity

    def add(self, experience):
        self.buffer.append(experience)  # Add an experience to the buffer

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # Randomly sample a batch of experiences

# Define the DQN agent that will interact with and learn from the environment
class DQNAgent:
    def __init__(self, env, buffer_capacity=100000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target network
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())  # Use Adam optimizer
        self.loss_fn = nn.MSELoss()  # Mean Squared Error Loss

    def select_action(self, state, epsilon):
        # Epsilon-greedy action selection
        available_columns = [col for col in range(7) if state[0][col] == 0]  # Check for valid moves
        if not available_columns:
            return None
        if random.random() < epsilon:
            return random.choice(available_columns)  # Choose random action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        q_values_clone = q_values.clone()
        q_values_clone[0, [col for col in range(7) if col not in available_columns]] = -float('inf')
        chosen_action = torch.argmax(q_values_clone).item()
        return chosen_action

    def train(self, num_episodes, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=0.9995):
        epsilon = epsilon_start
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            total_reward = 0
            for step in range(self.env.max_moves):
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add(Experience(state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                if done:
                    break
                if len(self.buffer.buffer) >= self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*experiences)
                    states = torch.stack(states)
                    next_states = torch.stack(next_states)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    dones = torch.tensor(dones, dtype=torch.float32)
                    q_values = self.model(states)
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states)
                        target_q_values = rewards + 0.99 * next_q_values.max(1)[0] * (1 - dones)
                    loss = self.loss_fn(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if step % self.target_update_frequency == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            epsilon = max(epsilon_final, epsilon * epsilon_decay)
            tqdm.write(f"Episode: {episode}, Total Reward: {total_reward:.4f}, Epsilon: {epsilon:.2f}")

# Execute the training process
if __name__ == '__main__':
    env = ConnectFourEnv()
    dqn_agent = DQNAgent(env)
    num_episodes = 10000
    dqn_agent.train(num_episodes=num_episodes)  # Train the agent
    torch.save({
        'model_state_dict': dqn_agent.model.state_dict(),
        'target_model_state_dict': dqn_agent.target_model.state_dict(),
        'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
    }, 'saved_agents/dqn_agent_after_training.pth')  # Save the model after training
