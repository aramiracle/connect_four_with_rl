import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from app.environment import ConnectFourEnv
from tqdm import tqdm

# Define the PPO model using PyTorch
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # Neural network layers for PPO
        self.fc1 = nn.Linear(6 * 7 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # Forward pass of the PPO model
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# PPO agent
class PPOAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, clip_param=0.2, entropy_coeff=0.01, gamma=0.99):
        # Proximal Policy Optimization (PPO) Agent
        self.env = env
        self.model = PPO()
        self.optimizer = optim.Adam(self.model.parameters())
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.entropy_coeff = entropy_coeff
        self.gamma = gamma
        self.buffer = []
        self.buffer_capacity = buffer_capacity

    def add_to_buffer(self, experience):
        # Add an experience tuple to the buffer
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_capacity:
            self.buffer = self.buffer[-self.buffer_capacity:]

    def select_action(self, state, training=True):
        # Select an action using the PPO model
        with torch.no_grad():
            policy_logits, _ = self.model(state)
            action_probs = torch.softmax(policy_logits, dim=1)

            valid_actions = self.env.get_valid_actions()

            if training:
                # During training, sample an action using multinomial from the valid actions
                valid_action_probs = action_probs[0, valid_actions]
                action_index = torch.multinomial(valid_action_probs + 1e-9, 1).item()
                action = valid_actions[action_index]
            else:
                # During testing, choose the action with the highest probability from the valid actions
                best_valid_action = torch.argmax(action_probs[0, valid_actions]).item()
                action = valid_actions[best_valid_action]

        return action


    def train_step(self):
        # Perform a single training step for the PPO model
        states, actions, old_probs, values, rewards, dones = zip(*self.buffer)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        returns = self.compute_returns(rewards, dones)

        for _ in range(3):  # Number of optimization epochs
            for i in range(0, len(self.buffer), self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_actions = actions[i:i + self.batch_size]
                batch_old_probs = old_probs[i:i + self.batch_size]
                batch_values = values[i:i + self.batch_size]
                batch_returns = returns[i:i + self.batch_size]

                logits, new_values = self.model(batch_states)
                new_probs = F.softmax(logits, dim=-1)
                new_action_probs = new_probs.gather(1, batch_actions.unsqueeze(-1))

                ratio = new_action_probs / batch_old_probs
                surr1 = ratio * (batch_returns - batch_values)
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * (batch_returns - batch_values)
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1).mean()

                loss = actor_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_returns(self, rewards, dones):
        # Compute the returns for the PPO training
        returns = []
        discounted_return = 0
        for i in range(len(rewards) - 1, -1, -1):
            discounted_return = rewards[i] + self.gamma * discounted_return * (1 - dones[i])
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns

    def reset_buffer(self):
        # Reset the buffer after each episode
        self.buffer = []


# Agent vs Agent Training using PPO
def agent_vs_agent_train_ppo(agents, env, num_episodes=1000):
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training (PPO)", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state)
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward
                old_probs, _ = agents[i].model(state.unsqueeze(0))
                agents[i].add_to_buffer((state, action, old_probs[0, action].item(), _, reward, done))
                state = next_state
                if done:
                    if env.winner == 1:
                        total_rewards[1] = -total_rewards[0]
                    elif env.winner == 2:
                        total_rewards[0] = -total_rewards[1]
                    break

        # Batch processing of experiences for each agent
        for agent in agents:
            agent.train_step()
            agent.reset_buffer()

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]:.4f}, Total Reward Player 2: {total_rewards[1]:.4f}")

if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    ppo_agents = [PPOAgent(env), PPOAgent(env)]

    # Agent vs Agent Training (PPO)
    agent_vs_agent_train_ppo(ppo_agents, env, num_episodes=100000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': ppo_agents[0].model.state_dict(),
        'optimizer_state_dict_player1': ppo_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': ppo_agents[1].model.state_dict(),
        'optimizer_state_dict_player2': ppo_agents[1].optimizer.state_dict(),
    }, 'saved_agents/ppo_agents_after_train.pth')