import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import namedtuple, deque
from tqdm import tqdm

# Define the Connect Four environment using Gym

class ConnectFourEnv(gym.Env):
    def __init__(self):
        self.board = None
        self.current_player = 1
        self.winner = None
        self.max_moves = 42  # Maximum number of moves in Connect Four
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.float32)

    def reset(self):
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1
        self.winner = None
        return self.board

    def step(self, action):
        if self.winner is not None:
            return self.board, 0, True, {}

        row = self.get_next_open_row(action)

        if row is not None:
            self.board[row, action] = self.current_player

            if self.check_win(row, action):
                self.winner = self.current_player
                reward = 1
                done = True
            elif np.count_nonzero(self.board) == self.max_moves:
                reward = 0
                done = True
            else:
                reward = 0
                done = False

            self.current_player = 3 - self.current_player  # Switch players
        else:
            # Handle the case where the column is already full
            reward = 0
            done = False

        return self.board, reward, done, {}

    def render(self):
        print(self.board)

    def get_next_open_row(self, col):
        for r in range(5, -1, -1):
            if self.board[r, col] == 0:
                return r

    def check_win(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.board[row, col]:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.board[row, col]:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

# Define the DQN model using PyTorch

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Implement experience replay buffer

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define the DQN agent

class DQNAgent:
    def __init__(self, env, buffer_capacity=10000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        # Find available columns (not full)
        available_columns = [col for col in range(7) if self.env.get_next_open_row(col) is not None]

        if not available_columns:
            # All columns are full, indicating a draw or game over
            return -1  # You can choose a special value to indicate a draw

        if random.random() < epsilon:
            return random.choice(available_columns)

        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            q_values = self.model(state)

        # Choose the action with the highest Q-value among available columns
        available_q_values = [q_values[0][col] for col in available_columns]
        return available_columns[torch.argmax(torch.stack(available_q_values)).item()]

    def train(self, num_episodes, epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=0.995):
        epsilon = epsilon_start

        for episode in tqdm(range(num_episodes)):  # Wrap the loop with tqdm
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

                    # Convert the list of numpy arrays to a single numpy array
                    states = np.array(states, dtype=np.single)
                    next_states = np.array(next_states, dtype=np.single)
                    rewards = np.array(rewards, dtype=np.single)
                    actions = np.array(actions, dtype=np.short)
                    dones = np.array(dones, dtype=np.single)

                    # Convert the numpy arrays to PyTorch tensors
                    states = torch.tensor(states, dtype=torch.float)
                    next_states = torch.tensor(next_states, dtype=torch.float)
                    rewards = torch.tensor(rewards, dtype=torch.float)
                    # Convert the 'actions' tensor to int64 data type
                    actions = torch.tensor(actions, dtype=torch.int64)
                    dones = torch.tensor(dones, dtype=torch.float)

                    q_values = self.model(states)
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states)
                        target_q_values = rewards + 0.99 * next_q_values.max(1)[0] * (1 - dones)

                    loss = self.loss_fn(q_values.gather(1, actions.view(-1, 1)), target_q_values.view(-1, 1))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if step % self.target_update_frequency == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            epsilon = max(epsilon_final, epsilon * epsilon_decay)
            tqdm.write(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")  # Use tqdm.write to update progress


# Main function
if __name__ == '__main__':
    env = ConnectFourEnv()
    dqn_agent = DQNAgent(env)

    # Train the DQN agent
    num_episodes = 30000
    dqn_agent.train(num_episodes=num_episodes)

   
    # Save the DQN agent's state after training
    torch.save({
        'model_state_dict': dqn_agent.model.state_dict(),
        'target_model_state_dict': dqn_agent.target_model.state_dict(),
        'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
    }, 'dqn_agent_after_training.pth')
