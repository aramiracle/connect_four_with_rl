import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
import numpy as np
from app.environment2 import ConnectFourEnv

# Define the C51 model using PyTorch
class C51(nn.Module):
    def __init__(self, num_atoms=51):
        super(C51, self).__init__()
        self.num_atoms = num_atoms
        self.fc1 = nn.Linear(6 * 7 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 7 * num_atoms)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape the output to (batch_size, 7, num_atoms) before applying softmax
        x = x.view(-1, 7, self.num_atoms)

        return F.softmax(x, dim=-1)

# Implement experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # Return the current size of the internal buffer
        return len(self.buffer)

# Update the DQNAgent to use the C51 model
class C51Agent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10, num_atoms=51, gamma=0.99):
        self.env = env
        self.model = C51(num_atoms)
        self.target_model = C51(num_atoms)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.num_training_steps = 0
        self.num_atoms = num_atoms
        self.gamma = gamma
        self.v_min = -10  # Minimum value for the distribution support
        self.v_max = 10   # Maximum value for the distribution support

    def select_action(self, state, epsilon):
        available_actions = self.env.get_valid_actions()
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                atom_probs = self.model(state_tensor).squeeze()
                q_values = (atom_probs * torch.linspace(self.v_min, self.v_max, self.num_atoms)).sum(dim=-1)
                masked_q_values = torch.full(q_values.shape, float('-inf'))
                masked_q_values[available_actions] = q_values[available_actions]
                action = torch.argmax(masked_q_values).item()

        self.model.train()
        return action

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            experiences = list(self.buffer.sample(self.batch_size))
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)  # Reshape actions tensor
            dones = torch.tensor(dones, dtype=torch.float32)

            current_atom_probs = self.model(states)
            current_atom_probs = current_atom_probs.gather(2, actions.unsqueeze(-1).expand(-1, -1, self.num_atoms))
            current_q_values = (current_atom_probs * torch.linspace(self.v_min, self.v_max, self.num_atoms)).sum(dim=-1)

            with torch.no_grad():
                next_atom_probs = self.target_model(next_states)
                next_q_values = (next_atom_probs * torch.linspace(self.v_min, self.v_max, self.num_atoms)).sum(dim=-1)
                max_next_q_values = torch.max(next_q_values, dim=-1).values
                target_distributions = self.project_distribution(rewards, 1 - dones, max_next_q_values)

            loss = self.compute_loss(current_atom_probs, target_distributions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

    def project_distribution(self, rewards, dones, values):
        batch_size = len(rewards)

        # Define the support for the distribution
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        # Compute the projected distribution
        projected_distribution = torch.zeros((batch_size, self.num_atoms))

        for i in range(batch_size):
            if dones[i]:
                # If the episode is done, project to a distribution concentrated at the final value
                index = int((values[i] - self.v_min) / delta_z)
                projected_distribution[i, index] += 1.0
            else:
                # If the episode is not done, apply the distributional Bellman update
                tz = rewards[i] + (1.0 - dones[i]) * support * self.gamma
                tz = torch.clamp(tz, self.v_min, self.v_max)
                b = (tz - self.v_min) / delta_z
                l, u = torch.floor(b).long(), torch.ceil(b).long()

                # Distribute probability mass to the lower and upper bounds
                projected_distribution[i, l] += support[u] * (u.float() - b)
                projected_distribution[i, u] += support[l] * (b - l.float())

        return projected_distribution

    def compute_loss(self, current_distribution, target_distribution):
        return self.loss_fn(torch.log(current_distribution), target_distribution)

def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.999):
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
                    break

        for agent in agents:
            agent.train_step()

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}, Epsilon: {epsilon:.2f}")

        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    c51_agents = [C51Agent(env), C51Agent(env)]

    # Agent vs Agent Training
    agent_vs_agent_train(c51_agents, env, num_episodes=30000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': c51_agents[0].model.state_dict(),
        'target_model_state_dict_player1': c51_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': c51_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': c51_agents[1].model.state_dict(),
        'target_model_state_dict_player2': c51_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': c51_agents[1].optimizer.state_dict(),
    }, 'saved_agents/c51_agents_after_train.pth')
