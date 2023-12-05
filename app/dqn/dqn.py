import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from app.environment2 import ConnectFourEnv

# Define the DQN model using PyTorch
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Neural network layers for the DQN model
        self.fc1 = nn.Linear(6 * 7 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 7)

    def forward(self, x):
        # Forward pass through the DQN model
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Implement experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        # Initialize the experience replay buffer with a given capacity
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
class DQNAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10):
        # Initialize the DQN agent with necessary parameters
        self.env = env
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0

    def select_action(self, state, epsilon):
        # Select an action using epsilon-greedy strategy
        available_actions = self.env.get_valid_actions()
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze()

            masked_q_values = torch.full(q_values.shape, float('-inf'))
            masked_q_values[available_actions] = q_values[available_actions]
            action = torch.argmax(masked_q_values).item()

        self.model.train()
        return action

    def train_step(self):
        # Perform a single training step using a batch of experiences
        if len(self.buffer) >= self.batch_size:
            experiences = list(self.buffer.sample(self.batch_size))
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                max_next_q_values = self.target_model(next_states).max(1)[0]
                expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1


def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.9999):
    # Train two DQN agents in an adversarial environment
    epsilon = epsilon_start

    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        states = [env.reset(), env.reset()]
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(states[i], epsilon)
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward
                agents[i].buffer.add(Experience(states[i], action, reward, next_state, done))
                states[i] = next_state
                if done:
                    if env.winner == 1:
                        total_rewards[1] = -total_rewards[0]
                    elif env.winner == 2:
                        total_rewards[0] = -total_rewards[1]
                    break

        for agent in agents:
            agent.train_step()

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]:.4f}, Total Reward Player 2: {total_rewards[1]:.4f}, Epsilon: {epsilon:.2f}")

        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    dqn_agents = [DQNAgent(env), DQNAgent(env)]

    # Agent vs Agent Training
    agent_vs_agent_train(dqn_agents, env, num_episodes=30000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': dqn_agents[0].model.state_dict(),
        'target_model_state_dict_player1': dqn_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': dqn_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': dqn_agents[1].model.state_dict(),
        'target_model_state_dict_player2': dqn_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': dqn_agents[1].optimizer.state_dict(),
    }, 'saved_agents/dqn_agents_after_train.pth')
