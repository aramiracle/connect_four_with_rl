import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from app.environment_train import ConnectFourEnv
import torch.nn.functional as F
from app.agents.alphazero.mcts import MonteCarloTreeSearch

# Define the neural network for AlphaZero
class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        super(AlphaZeroNetwork, self).__init__()
        # Modify the neural network architecture based on the ConnectFourEnv observations
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(6 * 7 * 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_policy = nn.Linear(128, 7)  # Output for policy head
        self.fc_value = nn.Linear(128, 1)  # Output for value head

    def forward(self, x):
        x = F.one_hot(x.long(), num_classes=3).float().permute(0, 3, 1, 2) # One-hot encode and reshape to channel first [N, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten for FC layers, use reshape instead of view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.fc_policy(x), dim=1)
        value = torch.tanh(self.fc_value(x))
        return policy, value

# Define the AlphaZero agent for ConnectFourEnv
class AlphaZeroAgent:
    def __init__(self, env, network, mcts_simulations=100, lr=0.001, c_puct=1.0): # Reduced simulations for faster training, added c_puct
        self.env = env
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.mcts = MonteCarloTreeSearch(self.network, simulations=mcts_simulations, c_puct=c_puct) # Pass c_puct to MCTS

    def select_action(self, env, use_mcts=True):
        state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0) # Convert board to tensor
        if use_mcts:
            # Use MCTS to select the best action
            action_probs = self.mcts.search(env)
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # During the test phase, simply use the current policy without MCTS
            with torch.no_grad():
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
        policy_loss = -torch.mean(torch.sum(policies * torch.log(predicted_policies + 1e-8), dim=1)) # Added small epsilon to log for numerical stability
        value_loss = nn.MSELoss()(predicted_values, values)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() # Return loss for monitoring


# AlphaZero vs AlphaZero training function for ConnectFourEnv
def alpha_zero_vs_alpha_zero_train(agent1, agent2, env, num_episodes=1000):
    history_agent1 = []
    history_agent2 = []
    reward_history_agent1 = [] # Track rewards for agent 1 (final game reward)
    reward_history_agent2 = [] # Track rewards for agent 2 (final game reward)

    for episode in tqdm(range(num_episodes), desc="AlphaZero vs AlphaZero Training", unit="episode"):
        states_agent1, policies_agent1, rewards_agent1 = [], [], [] # Added rewards lists (step rewards)
        states_agent2, policies_agent2, rewards_agent2 = [], [], [] # Added rewards lists (step rewards)
        env.reset()
        done = False
        current_player = 1 # Start with player 1

        while not done:
            if current_player == 1:
                # Agent 1's turn
                state_tensor_agent1 = torch.tensor(env.board, dtype=torch.float32) # Convert state to tensor
                states_agent1.append(state_tensor_agent1)
                policy1 = agent1.mcts.search(env)
                policies_agent1.append(policy1)

                action1 = agent1.select_action(env)
                next_state, reward1, done, _ = env.step(action1) # Get reward from step
                rewards_agent1.append(reward1) # Store step reward


            else:
                # Agent 2's turn
                state_tensor_agent2 = torch.tensor(env.board, dtype=torch.float32) # Convert state to tensor
                states_agent2.append(state_tensor_agent2)
                policy2 = agent2.mcts.search(env)
                policies_agent2.append(policy2)

                action2 = agent2.select_action(env)
                next_state, reward2, done, _ = env.step(action2) # Get reward from step
                rewards_agent2.append(reward2) # Store step reward


            current_player = 3 - current_player # Switch player (1 <-> 2)


        # Determine the final game outcome reward
        winner = "Unknown" # Initialize winner with a default value
        if env.winner == 1:
            winner = "Agent 1"
        elif env.winner == 2:
            winner = "Agent 2"
        else: # Draw or game ended without a winner (though in Connect Four, a game always ends with a winner or draw)
            winner = "Draw"

        # Use step rewards as values for training.
        values_agent1 = rewards_agent1
        values_agent2 = rewards_agent2


        episode_loss_agent1 = 0
        episode_loss_agent2 = 0

        # Train agents using the collected data
        if states_agent1:
            loss1 = agent1.train(states_agent1, policies_agent1, values_agent1)
            history_agent1.append(loss1)
            episode_loss_agent1 = loss1 # Store loss for printing
        if states_agent2:
            loss2 = agent2.train(states_agent2, policies_agent2, values_agent2)
            history_agent2.append(loss2)
            episode_loss_agent2 = loss2 # Store loss for printing

        avg_step_reward_agent1 = np.mean(rewards_agent1) if rewards_agent1 else 0
        avg_step_reward_agent2 = np.mean(rewards_agent2) if rewards_agent2 else 0

        reward_history_agent1.append(avg_step_reward_agent1)
        reward_history_agent2.append(avg_step_reward_agent2)

        tqdm.write(f"Episode {episode}: Winner: {winner}, Loss Agent 1: {episode_loss_agent1:.4f}, Loss Agent 2: {episode_loss_agent2:.4f}, Avg Reward Agent 1: {avg_step_reward_agent1:.2f}, Avg Reward Agent 2: {avg_step_reward_agent2:.2f}")
        if episode % 10 == 0: # Print average loss and reward every 10 episodes
            avg_loss1 = np.mean(history_agent1[-10:]) if history_agent1 else 0
            avg_loss2 = np.mean(history_agent2[-10:]) if history_agent2 else 0
            avg_reward_agent1 = np.mean(reward_history_agent1[-10:]) if reward_history_agent1 else 0
            avg_reward_agent2 = np.mean(reward_history_agent2[-10:]) if reward_history_agent2 else 0
            tqdm.write(f"Episode {episode-10}-{episode}: Avg Loss Agent 1: {avg_loss1:.4f}, Avg Loss Agent 2: {avg_loss2:.4f}, Avg Reward Agent 1: {avg_reward_agent1:.2f}, Avg Reward Agent 2: {avg_reward_agent2:.2f}")


    return history_agent1, history_agent2, reward_history_agent1, reward_history_agent2 # reward_history is now used to track final reward


if __name__ == '__main__':
    env = ConnectFourEnv()

    # Create two instances of the AlphaZero network
    az_network1 = AlphaZeroNetwork()
    az_network2 = AlphaZeroNetwork()

    # Create two instances of the AlphaZero agent for ConnectFourEnv
    az_agent1 = AlphaZeroAgent(env, az_network1, mcts_simulations=10) # Reduced simulations for faster training
    az_agent2 = AlphaZeroAgent(env, az_network2, mcts_simulations=10)

    # AlphaZero vs AlphaZero Training for ConnectFourEnv
    history_agent1, history_agent2, reward_history_agent1, reward_history_agent2 = alpha_zero_vs_alpha_zero_train(az_agent1, az_agent2, env, num_episodes=1000) # Reduced episodes for testing

    # Save the trained AlphaZero agents
    torch.save({
        'model_state_dict_agent1': az_agent1.network.state_dict(),
        'optimizer_state_dict_agent1': az_agent1.optimizer.state_dict(),
        'model_state_dict_agent2': az_agent2.network.state_dict(),
        'optimizer_state_dict_agent2': az_agent2.optimizer.state_dict(),
        'training_history_agent1': history_agent1,
        'training_history_agent2': history_agent2,
        'reward_history_agent1': reward_history_agent1,
        'reward_history_agent2': reward_history_agent2,

    }, 'saved_agents/alpha_zero_agents_after_train.pth') # Changed save name to indicate final reward usage

    print("Training complete and agents saved to 'saved_agents/alpha_zero_agents_after_train.pth'") # Changed save name in print