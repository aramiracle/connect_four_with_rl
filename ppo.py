import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import ConnectFourEnv
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 7 * 3)        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size=32, learning_rate=0.0001, gamma=0.99, clip_param=0.1, epochs=10):
        self.policy = PolicyNetwork(state_dim, hidden_size, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_param = clip_param
        self.epochs = epochs

    def select_action(self, state):
        state = state.flatten().float().unsqueeze(0).to(device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, states, actions, old_probs, rewards, dones):
        returns = self.compute_returns(rewards, dones)
        states = torch.cat(states).to(device)  # Convert list of tensors to a single tensor
        actions, old_probs, returns = map(torch.tensor, (actions, old_probs, returns))

        for _ in range(self.epochs):
            advantages = returns - self.policy(states).gather(1, actions.unsqueeze(1))

            new_probs = self.policy(states)
            ratio = (new_probs.gather(1, actions.unsqueeze(1)) / old_probs).squeeze()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            loss = -torch.min(surr1, surr2).mean()

            entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=1).mean()
            loss -= 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R if not done else 0
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns.numpy()

def train_ppo(env, num_episodes=1000, save_path="saved_agents/ppo_agent_after_train.pth"):
    action_dim = env.action_space.n
    flattened_state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    ppo = PPOAgent(flattened_state_size, action_dim)

    for episode in tqdm(range(num_episodes)):
        state = env.reset().view(1, 6 * 7)
        done = False
        total_rewards = 0
        states, actions, rewards, dones, old_probs = [], [], [], [], []

        while not done:
            action, log_prob = ppo.select_action(state)
            states.append(state)
            actions.append(action)
            old_probs.append(log_prob.item())

            state, reward, done, _ = env.step(action)
            state = state.view(1, 6 * 7)
            rewards.append(reward)
            dones.append(done)

            total_rewards += reward

        ppo.update(states, actions, old_probs, rewards, dones)

        if episode % 10 == 0:
            tqdm.write(f"Episode {episode}, Total Reward {total_rewards:.4f}")

        if episode % 100 == 0:
            save_model(ppo, save_path)

def save_model(model, save_path):
    torch.save({
        'model_state_dict': model.policy.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, save_path)


def agent_vs_agent_train(agents, env, num_episodes=1000):
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        state = env.reset()
        done = False
        total_rewards = [0, 0]
        while not done:
            for i in range(len(agents)):
                agent = agents[i]
                state = state.view(1, 6 * 7)
                action_probs = agent.policy(state)
                action = torch.argmax(action_probs).item()
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward
                state = next_state
                if done:
                    # Adjust total_rewards if there is a winner
                    if env.winner == 1:
                        total_rewards[1] = -total_rewards[0]
                    elif env.winner == 2:
                        total_rewards[0] = -total_rewards[1]
                    else:
                        total_rewards = [0, 0]
                    break
        winner = info.get("winner", None)
        tqdm.write(f"Episode {episode}, Winner: {winner}, Total Reward Player 1: {total_rewards[0]:.4f}, Total Reward Player 2: {total_rewards[1]:.4f}")

    env.close()


# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Player
    ppo_agent = PPOAgent(6 * 7, 7)

    # PPO Training
    train_ppo(env, num_episodes=1000, save_path='saved_agents/single_ppo_agent_after_train.pth')

    # Load PPO Agent
    checkpoint = torch.load('saved_agents/single_ppo_agent_after_train.pth')
    ppo_agent.policy.load_state_dict(checkpoint['model_state_dict'])
    ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Create two PPO agents from the saved one
    ppo_agents = [PPOAgent(6 * 7, 7), PPOAgent(6 * 7, 7)]
    ppo_agents[0].policy.load_state_dict(ppo_agent.policy.state_dict())
    ppo_agents[1].policy.load_state_dict(ppo_agent.policy.state_dict())
    ppo_agents[0].optimizer.load_state_dict(ppo_agent.optimizer.state_dict())
    ppo_agents[1].optimizer.load_state_dict(ppo_agent.optimizer.state_dict())

    # Agent vs Agent Training
    agent_vs_agent_train(ppo_agents, env, num_episodes=10000)

    torch.save({
        'model_state_dict_player1': ppo_agents[0].policy.state_dict(),
        'optimizer_state_dict_player1': ppo_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': ppo_agents[1].policy.state_dict(),
        'optimizer_state_dict_player2': ppo_agents[1].optimizer.state_dict(),
    }, 'saved_agents/ppo_agents_after_train.pth')
