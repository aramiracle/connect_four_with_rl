import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from app.environment2 import ConnectFourEnv
import torch.nn.functional as F
from app.alphazero.mcts import MonteCarloTreeSearch

# Define the neural network for AlphaZero
class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        super(AlphaZeroNetwork, self).__init__()
        # Modify the neural network architecture based on the ConnectFourEnv observations
        self.fc1 = nn.Linear(6 * 7 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_policy = nn.Linear(128, 7)  # Output for policy head
        self.fc3_value = nn.Linear(128, 1)  # Output for value head

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.fc3_policy(x), dim=1)
        value = torch.tanh(self.fc3_value(x))
        return policy, value

# Define the AlphaZero agent for ConnectFourEnv
class AlphaZeroAgent:
    def __init__(self, env, network, mcts_simulations=1000, lr=0.001):
        self.env = env
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.mcts = MonteCarloTreeSearch(self.network, simulations=mcts_simulations)

    def select_action(self, env, use_mcts=True):
        if use_mcts:
            # Use MCTS to select the best action
            action_probs = self.mcts.search(env)
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # During the test phase, simply use the current policy without MCTS
            state_tensor = torch.tensor(env.board, dtype=torch.float32)
            action_probs, _ = self.network(state_tensor)
            action = torch.argmax(action_probs).item()

        return action

    def train(self, states, policies, values):
        # Convert data to PyTorch tensors
        states = torch.stack(states)
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32).view(-1, 1)

        # Forward pass
        predicted_policies, predicted_values = self.network(states)

        # Compute losses
        policy_loss = -torch.sum(policies * torch.log(predicted_policies), dim=1).mean()
        value_loss = nn.MSELoss()(predicted_values, values)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# AlphaZero training function for ConnectFourEnv
def alpha_zero_train(agent, env, num_episodes=1000):
    for episode in tqdm(range(num_episodes), desc="AlphaZero Training", unit="episode"):
        states, policies, values = [], [], []
        env.reset()
        done = False

        while not done:
            action_probs = agent.mcts.search(env)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, _, done, _ = env.step(action)

            # Store the state, policy, and value for training
            states.append(env.board)
            policies.append(action_probs)
            values.append(0.5)  # Placeholder for demonstration, update with actual values

            state = next_state

        # Train the agent using the collected data
        agent.train(states, policies, values)

if __name__ == '__main__':
    env = ConnectFourEnv()

    # Create an instance of the AlphaZero network
    az_network = AlphaZeroNetwork()

    # Create an instance of the AlphaZero agent for ConnectFourEnv
    az_agent = AlphaZeroAgent(env, az_network, mcts_simulations=20)

    # AlphaZero Training for ConnectFourEnv
    alpha_zero_train(az_agent, env, num_episodes=3000)

    # Save the trained AlphaZero agent
    torch.save({
        'model_state_dict': az_agent.network.state_dict(),
        'optimizer_state_dict': az_agent.optimizer.state_dict(),
    }, 'saved_agents/alpha_zero_agent_after_train.pth')
