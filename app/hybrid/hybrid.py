import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from app.environment2 import ConnectFourEnv

# Define the Dueling DQN model using PyTorch
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Value stream layers
        self.fc1_value = nn.Linear(6 * 7 * 3, 256)
        self.fc2_value = nn.Linear(256, 128)
        self.fc3_value = nn.Linear(128, 64)
        self.fc4_value = nn.Linear(64, 1)

        # Advantage stream layers
        self.fc1_advantage = nn.Linear(6 * 7 * 3, 256)
        self.fc2_advantage = nn.Linear(256, 128)
        self.fc3_advantage = nn.Linear(128, 64)
        self.fc4_advantage = nn.Linear(64, 7)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)

        # Value stream
        x_value = F.relu(self.fc1_value(x))
        x_value = F.relu(self.fc2_value(x_value))
        x_value = F.relu(self.fc3_value(x_value))
        value = self.fc4_value(x_value)

        # Advantage stream
        x_advantage = F.relu(self.fc1_advantage(x))
        x_advantage = F.relu(self.fc2_advantage(x_advantage))
        x_advantage = F.relu(self.fc3_advantage(x_advantage))
        advantage = self.fc4_advantage(x_advantage)

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
class HybridAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DuelingDQN()  # Change here
        self.target_model = DuelingDQN()  # Change here
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
                    state_tensor = state.unsqueeze(0)  # Adding batch dimension
                    with torch.no_grad():
                        q_values = self.model(state_tensor).squeeze()

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
        if next_env.winner is not None:  # Update this line based on your winner identification
            return True
        return False
    
    def is_instant_loss(self, env, action):
        # Check if the opponent has an instant winning move in the next turn
        next_env = env.clone()
        # print(next_env.current_player)
        next_env.step(action)
        opponent_valid_actions = next_env.get_valid_actions()
        for opponent_action in opponent_valid_actions:
            opponent_next_env = next_env.clone()
            opponent_next_env.step(opponent_action)
            if opponent_next_env.winner is not None:  # Update this line
                return True

        return False

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
            
            # Check if the chosen action is an instant loss move
            instant_loss_mask = torch.tensor([self.is_instant_loss(self.env, a) for a in actions], dtype=torch.float32)
            
            # Introduce a penalty term for instant loss moves
            penalty_factor = 0.5  # You can adjust the penalty factor
            penalty = penalty_factor * instant_loss_mask

            # Calculate the expected Q values with the penalty
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values - penalty  # Assuming a gamma of 0.99
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.999):
    epsilon = epsilon_start

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

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}, Epsilon: {epsilon:.2f}")

        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    hybrid_agents = [HybridAgent(env), HybridAgent(env)]

    # Agent vs Agent Training
    agent_vs_agent_train(hybrid_agents, env, num_episodes=100000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': hybrid_agents[0].model.state_dict(),
        'target_model_state_dict_player1': hybrid_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': hybrid_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': hybrid_agents[1].model.state_dict(),
        'target_model_state_dict_player2': hybrid_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': hybrid_agents[1].optimizer.state_dict(),
    }, 'saved_agents/hybrid_agents_after_train.pth')
