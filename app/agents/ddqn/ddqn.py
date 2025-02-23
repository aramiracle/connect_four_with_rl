import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from app.environment_train import ConnectFourEnv

from typing import List

# Define the Dueling DQN model using PyTorch
class ConvDuelingDQN(nn.Module):
    def __init__(self):
        super(ConvDuelingDQN, self).__init__()
        # Convolutional layers for feature extraction - Reduced channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Reduced to 32 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Added conv3 with 64 channels

        # Value stream layers - Reduced size, Adjusted input size for FC layers
        self.fc_value = nn.Linear(6 * 7 * 64, 128) # Adjusted input size to 6*7*64 due to conv3
        self.fc1_value = nn.Linear(128, 32) # Reduced FC size
        self.fc2_value = nn.Linear(32, 1)

        # Advantage stream layers - Reduced size, Adjusted input size for FC layers
        self.fc_advantage = nn.Linear(6 * 7 * 64, 128) # Adjusted input size to 6*7*64 due to conv3
        self.fc1_advantage = nn.Linear(128, 32) # Reduced FC size
        self.fc2_advantage = nn.Linear(32, 7)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.unsqueeze(0) if len(x.size())==3 else x
        x = x.permute(0, 3, 1, 2) # Reshape to (batch_size, channels, width, height)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # Added conv3

        x = x.reshape(x.size(0), -1) # Flatten for FC layers

        # Value stream
        x_value = F.relu(self.fc_value(x))
        x_value = F.relu(self.fc1_value(x_value))
        value = self.fc2_value(x_value)

        # Advantage stream
        x_advantage = F.relu(self.fc_advantage(x))
        x_advantage = F.relu(self.fc1_advantage(x_advantage))
        advantage = self.fc2_advantage(x_advantage)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Implement experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        # Add an experience to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # Return the current size of the internal buffer
        return len(self.buffer)

# Define the DQN agent
class DDQNAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = ConvDuelingDQN()  # Change here
        self.target_model = ConvDuelingDQN()  # Change here
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0

    def select_action(self, state, epsilon):
        # Directly check which columns are not full
        available_actions = self.env.get_valid_actions()

        # Ensure the model is in evaluation mode
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            state_tensor = state.unsqueeze(0)  # Adding batch dimension
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze()

            # Mask the Q-values of invalid actions with a very negative number
            masked_q_values = torch.full(q_values.shape, float('-inf'))
            masked_q_values[available_actions] = q_values[available_actions]

            # Get the action with the highest Q-value among the valid actions
            action = torch.argmax(masked_q_values).item()

        # Ensure the model is back in training mode
        self.model.train()

        return action

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            
            experiences = list(self.buffer.sample(self.batch_size))  # Convert to list for better indexing
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Use target model for action selection in Double Q-learning
            target_actions = self.model(next_states).max(1)[1].unsqueeze(-1)
            max_next_q_values = self.target_model(next_states).gather(1, target_actions).squeeze(-1)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values  # Assuming a gamma of 0.99
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1
        pass

def agent_vs_agent_train(agents, env, num_episodes=200000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.9999):
    epsilon = epsilon_start
    reward_history_p1 = []
    reward_history_p2 = []

    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state, epsilon)
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward
                agents[i].buffer.add(Experience(state, action, reward, next_state, done))
                state = next_state
                if done:
                    break

        # Batch processing of experiences for each agent
        for agent in agents:
            agent.train_step()

        reward_history_p1.append(total_rewards[0])
        reward_history_p2.append(total_rewards[1])

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}, Epsilon: {epsilon:.2f}")

        if episode % 100 == 0 and episode > 0:
            avg_reward_p1 = sum(reward_history_p1[-100:]) / 100
            avg_reward_p2 = sum(reward_history_p2[-100:]) / 100
            tqdm.write(f"--- Episode {episode} Report ---")
            tqdm.write(f"Average Reward over last 100 episodes - Player 1: {avg_reward_p1:.2f}, Player 2: {avg_reward_p2:.2f}")
            tqdm.write(f"------------------------------")


        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    dqn_agents = [DDQNAgent(env), DDQNAgent(env)]

    # Agent vs Agent Training
    agent_vs_agent_train(dqn_agents, env, num_episodes=100000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': dqn_agents[0].model.state_dict(),
        'target_model_state_dict_player1': dqn_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': dqn_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': dqn_agents[1].model.state_dict(),
        'target_model_state_dict_player2': dqn_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': dqn_agents[1].optimizer.state_dict(),
    }, 'saved_agents/ddqnd_agents_after_train.pth')
