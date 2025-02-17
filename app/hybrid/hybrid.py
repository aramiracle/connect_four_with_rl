import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from app.environment_train import ConnectFourEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Convolutional Dueling DQN model using PyTorch - Smaller Version with one more layer
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

# Implement experience replay buffer (No changes needed)
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

# Pure Python is_instant_win and is_instant_loss
def is_instant_win(board):
    rows, cols = board.shape
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3] == 1:
                return True
    # Check vertical
    for r in range(rows - 3):
        for c in range(cols):
            if board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c] == 1:
                return True
    # Check diagonals (top-left to bottom-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3] == 1:
                return True
    # Check diagonals (top-right to bottom-left)
    for r in range(rows - 3):
        for c in range(3, cols):
            if board[r, c] == board[r + 1, c - 1] == board[r + 2, c - 2] == board[r + 3, c - 3] == 1:
                return True
    return False

def is_instant_loss(board):
    rows, cols = board.shape
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3] == 2:
                return True
    # Check vertical
    for r in range(rows - 3):
        for c in range(cols):
            if board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c] == 2:
                return True
    # Check diagonals (top-left to bottom-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3] == 2:
                return True
    # Check diagonals (top-right to bottom-left)
    for r in range(rows - 3):
        for c in range(3, cols):
            if board[r, c] == board[r + 1, c - 1] == board[r + 2, c - 2] == board[r + 3, c - 3] == 2:
                return True
    return False

# Define the DQN agent
class HybridAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=128, target_update_frequency=500, learning_rate=0.0001, instant_loss_penalty=1.0, use_small_model=True): # Added use_small_model flag
        self.env = env
        if use_small_model:
            self.model = ConvDuelingDQN().to(device)  # Using smaller ConvDuelingDQN model with one more layer
            self.target_model = ConvDuelingDQN().to(device)  # Using smaller ConvDuelingDQN model with one more layer
        else:
            self.model = ConvDuelingDQN().to(device)  # Using original ConvDuelingDQN model
            self.target_model = ConvDuelingDQN().to(device)  # Using original ConvDuelingDQN model

        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # Adjusted learning rate
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0
        self.instant_loss_penalty_factor = instant_loss_penalty # Penalty factor is now a class attribute

    def select_action(self, state, epsilon):
        # Directly check which columns are not full
        available_actions = self.env.get_valid_actions()

        # Ensure the model is in evaluation mode
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            # Check for instant win moves
            instant_win_actions = [action for action in available_actions if self.is_instant_win(self.env, action)]
            if instant_win_actions:
                # If there are instant win moves, choose one randomly
                action = random.choice(instant_win_actions)
            else:
                # Filter out instant loss moves
                filtered_actions = [action for action in available_actions if not self.is_instant_loss(self.env, action)]
                if filtered_actions:
                    # If there are filtered actions, choose the action with the highest Q-value among them
                    state_tensor = state.unsqueeze(0).to(device)  # Adding batch dimension, move state to device
                    with torch.no_grad():
                        q_values = self.model(state_tensor).squeeze().cpu() # Move output back to CPU for action selection (argmax)

                    # Mask the Q-values of invalid actions with a very negative number
                    masked_q_values = torch.full(q_values.shape, float('-inf'))
                    masked_q_values[filtered_actions] = q_values[filtered_actions]

                    # Get the action with the highest Q-value among the valid actions
                    action = torch.argmax(masked_q_values).item()
                else:
                    # If there are no filtered actions, choose a random action from all available actions
                    action = random.choice(available_actions)

        # Ensure the model is back in training mode
        self.model.train()

        return action

    def is_instant_win(self, env, action):
        # Check if the agent has an instant winning move in the next turn
        next_env = env.clone()
        next_env.step(action)
        # Convert PyTorch tensor board to NumPy array before passing to Numba
        board_np = next_env.board.cpu().numpy()
        return is_instant_win(board_np)

    def is_instant_loss(self, env, action):
        # Check if the opponent has an instant winning move in the next turn
        next_env = env.clone()
        next_env.step(action)
        opponent_valid_actions = next_env.get_valid_actions()
        for opponent_action in opponent_valid_actions:
            opponent_next_env = next_env.clone()
            opponent_next_env.step(opponent_action)
            # Convert PyTorch tensor board to NumPy array before passing to Numba
            board_np = opponent_next_env.board.cpu().numpy()
            if is_instant_loss(board_np):
                return True
        return False

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            experiences = list(self.buffer.sample(self.batch_size))  # Convert to list for better indexing
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states).to(device) # Move states to device
            next_states = torch.stack(next_states).to(device) # Move next_states to device
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device) # Move rewards to device
            actions = torch.tensor(actions, dtype=torch.int64).to(device) # Move actions to device
            dones = torch.tensor(dones, dtype=torch.float32).to(device) # Move dones to device

            # Use target model for action selection in Double Q-learning
            target_actions = self.model(next_states).max(1)[1].unsqueeze(-1)
            max_next_q_values = self.target_model(next_states).gather(1, target_actions).squeeze(-1)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Check if the chosen action is an instant loss move
            instant_loss_mask_list = []
            temp_env = ConnectFourEnv() # Need to create a new env for each state to check instant loss independently
            for i in range(len(states)):
                temp_env.reset(states[i].cpu().numpy()) # Set the board to the state from experience buffer, keep on CPU as env is likely CPU-based
                instant_loss_mask_list.append(self.is_instant_loss(temp_env, actions[i].item()))
            instant_loss_mask = torch.tensor(instant_loss_mask_list, dtype=torch.float32).to(device) # Move mask to device


            # Introduce a penalty term for instant loss moves
            penalty = self.instant_loss_penalty_factor * instant_loss_mask

            # Calculate the expected Q values with the penalty
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values - penalty  # Assuming a gamma of 0.99
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

def agent_vs_agent_train(agents, env, num_episodes=200000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.9995): # Increased episodes, adjusted decay
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

        if episode % 1000 == 0 and episode > 0:
            avg_reward_p1 = sum(reward_history_p1[-1000:]) / 1000
            avg_reward_p2 = sum(reward_history_p2[-1000:]) / 1000
            tqdm.write(f"--- Episode {episode} Report ---")
            tqdm.write(f"Average Reward over last 1000 episodes - Player 1: {avg_reward_p1:.2f}, Player 2: {avg_reward_p2:.2f}")
            tqdm.write(f"------------------------------")


        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players - using smaller model with one more layer
    hybrid_agents = [HybridAgent(env, use_small_model=True), HybridAgent(env, use_small_model=True)]

    # Agent vs Agent Training
    agent_vs_agent_train(hybrid_agents, env, num_episodes=10000) # Increased episodes

    # Save the trained agents - saving smaller agents with one more layer
    torch.save({
        'model_state_dict_player1': hybrid_agents[0].model.state_dict(),
        'target_model_state_dict_player1': hybrid_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': hybrid_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': hybrid_agents[1].model.state_dict(),
        'target_model_state_dict_player2': hybrid_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': hybrid_agents[1].optimizer.state_dict(),
    }, 'saved_agents/hybrid_agents_after_train.pth')