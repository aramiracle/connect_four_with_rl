import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment import ConnectFourEnv
from tqdm import tqdm

# Define a simple neural network for policy and value estimation
class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_policy = nn.Linear(64, 7)  # Output for policy
        self.fc_value = nn.Linear(64, 1)   # Output for value

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

# A3C agent implementation
class A3CAgent:
    def __init__(self, env, num_workers, gamma=0.99, lr=1e-3):
        self.env = env
        self.num_workers = num_workers
        self.gamma = gamma
        self.lr = lr
        self.loss_fn = nn.MSELoss()

        self.model = PolicyValueNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, rollout):
        states, actions, rewards, next_states, dones, values = rollout
        next_values = self.model(next_states)[1].detach()

        returns = self.compute_returns(rewards, next_values, dones)

        policy_logits, value = self.model(states)
        action_probs = torch.softmax(policy_logits, dim=1)
        m = Categorical(action_probs)
        entropy = m.entropy().mean()

        advantage = returns - value.squeeze(1)
        policy_loss = -m.log_prob(actions) * advantage.detach()
        value_loss = self.loss_fn(value.squeeze(1), returns.detach())

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.mean().backward()
        self.optimizer.step()

    def compute_returns(self, rewards, next_values, dones):
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        returns[-1] = rewards[-1] + (1 - dones[-1]) * self.gamma * next_values[-1]

        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        return returns

    def run_episode(self):
        rollout = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': []
        }

        state = self.env.reset()

        while True:
            policy_logits, value = self.model(torch.tensor(state, dtype=torch.float32))
            action_probs = torch.softmax(policy_logits, dim=0)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = self.env.step(action)

            rollout['states'].append(state)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_states'].append(next_state)
            rollout['dones'].append(done)
            rollout['values'].append(value.item())

            state = next_state

            if done:
                break

        return rollout

    def train_async(self, num_episodes=1000):
        for _ in tqdm(range(num_episodes), desc='Training'):
            rollouts = []

            for _ in range(self.num_workers):
                rollout = self.run_episode()
                rollouts.append(rollout)

            for rollout in rollouts:
                self.train(self.unpack_rollout(rollout))

    def unpack_rollout(self, rollout):
        states = torch.stack(rollout['states'])
        actions = torch.tensor(rollout['actions'], dtype=torch.int64)
        rewards = torch.tensor(rollout['rewards'], dtype=torch.float32)
        next_states = torch.stack(rollout['next_states'])
        dones = torch.tensor(rollout['dones'], dtype=torch.float32)
        values = torch.tensor(rollout['values'], dtype=torch.float32)
        return states, actions, rewards, next_states, dones, values

if __name__=='__main__':
    # Example usage:
    env = ConnectFourEnv()
    agent = A3CAgent(env, num_workers=4)
    agent.train_async(num_episodes=10000)

    # Save the trained model
    torch.save(agent.model.state_dict(), 'saved_agents/a3c_agent_after_train.pth')