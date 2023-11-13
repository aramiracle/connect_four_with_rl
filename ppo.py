import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import ConnectFourEnv

# Neural network architecture for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Proximal Policy Optimization algorithm
class PPO:
    def __init__(self, state_dim, action_dim, hidden_size=64, learning_rate=0.001, gamma=0.99, clip_param=0.2, epochs=10):
        self.policy = PolicyNetwork(state_dim, hidden_size, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_param = clip_param
        self.epochs = epochs

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, states, actions, old_probs, rewards, dones):
        returns = self.compute_returns(rewards, dones)

        # Convert everything to PyTorch tensors
        states = torch.from_numpy(states).float()
        actions = torch.tensor(actions)
        old_probs = torch.tensor(old_probs)
        returns = torch.tensor(returns)

        for _ in range(self.epochs):
            # Calculate advantages
            advantages = returns - self.policy(states).gather(1, actions.unsqueeze(1))

            # Calculate surrogate loss
            new_probs = self.policy(states)
            ratio = (new_probs.gather(1, actions.unsqueeze(1)) / old_probs).squeeze()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            loss = -torch.min(surr1, surr2).mean()

            # Optimize policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns.numpy()

# Training the PPO agent
def train_ppo(env, num_episodes=1000):
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        states, actions, rewards, dones, old_probs = [], [], [], [], []

        while not done:
            action, log_prob = ppo.select_action(state)
            states.append(state.flatten())
            actions.append(action)
            old_probs.append(log_prob.item())

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)

            total_reward += reward

        ppo.update(states, actions, old_probs, rewards, dones)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__=='__main__':
    train_ppo(ConnectFourEnv(), num_episodes=1000)