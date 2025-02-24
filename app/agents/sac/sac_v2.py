import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
from tqdm import tqdm

# ===========================
# Prioritized Replay Buffer
# ===========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        # When adding a new transition, use the maximum priority so far.
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.stack([torch.tensor(item[3], dtype=torch.float32) for item in batch])
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(-1)
        return states, actions, rewards, next_states, dones, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error.item()) + 1e-5)

    def __len__(self):
        return len(self.buffer)

# ===========================
# Actor Network (Policy)
# ===========================
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        # Input: Connect Four board as a (6,7) grid with values {0,1,2}.
        # We one-hot encode to 3 channels.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.fc_policy = nn.Linear(128, 7)  # 7 possible moves (columns)

    def forward(self, x):
        # x shape: (batch, 6, 7) with integer entries 0,1,2.
        x = x.long()
        x = F.one_hot(x, num_classes=3).float()  # shape: (batch, 6, 7, 3)
        x = x.permute(0, 3, 1, 2)  # shape: (batch, 3, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        return policy_logits

# ===========================
# Critic Network (Q-value)
# ===========================
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        # Process the board state (6,7) as above.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # After convolution: 64 * 6 * 7 = 2688 features.
        # The action is one-hot encoded (7 dims).
        self.fc1 = nn.Linear(2688 + 7, 128)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, state, action):
        # state: (batch, 6, 7), action: (batch, 1)
        state = state.long()
        state = F.one_hot(state, num_classes=3).float()  # (batch, 6, 7, 3)
        state = state.permute(0, 3, 1, 2)  # (batch, 3, 6, 7)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        action_onehot = F.one_hot(action.squeeze(-1), num_classes=7).float()
        x = torch.cat([x, action_onehot], dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        return value

# ===========================
# Soft Actor-Critic Agent
# ===========================
class SACAgent:
    def __init__(self, env, gamma=0.99, lr=1e-4, init_alpha=0.2, tau=0.005, buffer_size=1000000, batch_size=16):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size

        # Automatic entropy tuning parameters:
        self.target_entropy = -np.log(1/7)  # target entropy for 7 actions
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = init_alpha

        # Initialize networks:
        self.actor = ActorNet()
        self.critic1 = CriticNet()
        self.critic2 = CriticNet()
        self.critic1_target = CriticNet()
        self.critic2_target = CriticNet()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
    
    def select_action(self, state, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state_tensor)
        action_probs = torch.softmax(logits, dim=1)
        valid_actions = self.env.get_valid_actions()  # should return list of valid column indices
        if training:
            valid_probs = action_probs[0][valid_actions]
            if valid_probs.sum().item() == 0:
                action = random.choice(valid_actions)
            else:
                valid_probs = valid_probs / valid_probs.sum()
                m = Categorical(probs=valid_probs)
                action_index = m.sample().item()
                action = valid_actions[action_index]
        else:
            valid_probs = action_probs[0][valid_actions]
            action_index = torch.argmax(valid_probs).item()
            action = valid_actions[action_index]
        return action

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(self.batch_size)

        # --- Critic Update ---
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=1)
            next_dist = Categorical(probs=next_probs)
            next_actions = next_dist.sample()
            log_prob_next = next_dist.log_prob(next_actions).unsqueeze(-1)
            target_q1 = self.critic1_target(next_states, next_actions.unsqueeze(-1))
            target_q2 = self.critic2_target(next_states, next_actions.unsqueeze(-1))
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * log_prob_next)
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor Update ---
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs=probs)
        sampled_actions = dist.sample()
        log_prob_actions = dist.log_prob(sampled_actions).unsqueeze(-1)
        q1_val = self.critic1(states, sampled_actions.unsqueeze(-1))
        q2_val = self.critic2(states, sampled_actions.unsqueeze(-1))
        q_val = torch.min(q1_val, q2_val)
        actor_loss = (self.alpha * log_prob_actions - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Automatic Entropy Tuning ---
        alpha_loss = -(self.log_alpha * (log_prob_actions + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Soft Update Target Networks ---
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        # --- Update Replay Priorities ---
        td_errors = (current_q1 - target).detach().abs().squeeze()
        self.replay_buffer.update_priorities(indices, td_errors)

    def run_episode(self, training=True):
        state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state, training=training)
            next_state, reward, done, info = self.env.step(action)
            reward = reward if reward is not None else 0
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
        return info

# ===========================
# Self-Play Training Function
# ===========================
def agent_vs_agent_train_sac_v2(agents, env, num_episodes=1000, batch_size=16):
    win_counts = [0, 0, 0]  # [Player 1 wins, Player 2 wins, Draws]
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        player_turn = 0  # Start with agent 0
        while not done:
            agent = agents[player_turn]
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            r = reward if reward is not None else 0
            # Store the same transition for both agents
            for ag in agents:
                ag.replay_buffer.push(state, action, r, next_state, done)
            state = next_state
            player_turn = 1 - player_turn
        # Update win counts based on info (assumes info['winner'] is set)
        if info.get('winner') == 'Player 1':
            win_counts[0] += 1
        elif info.get('winner') == 'Player 2':
            win_counts[1] += 1
        else:
            win_counts[2] += 1

        # Train both agents
        for ag in agents:
            ag.train()

        total_games = sum(win_counts)
        if total_games > 0 and episode % 100 == 0:
            win_rate_p1 = win_counts[0] / total_games
            win_rate_p2 = win_counts[1] / total_games
            draw_rate = win_counts[2] / total_games
            tqdm.write(f"Episode {episode}: P1 win rate: {win_rate_p1:.3f}, P2 win rate: {win_rate_p2:.3f}, Draw: {draw_rate:.3f}")
    env.close()

# ===========================
# Main: Train and Save Models
# ===========================
if __name__ == '__main__':
    from app.environment_test import ConnectFourEnv  # Ensure your ConnectFourEnv is implemented
    env = ConnectFourEnv()
    agent1 = SACAgent(env, batch_size=16)
    agent2 = SACAgent(env, batch_size=16)
    agents = [agent1, agent2]
    
    # Train for a specified number of episodes (e.g., 30,000)
    agent_vs_agent_train_sac_v2(agents, env, num_episodes=30000, batch_size=16)

    # Save the trained models
    torch.save({
        'actor_state_dict_player1': agent1.actor.state_dict(),
        'critic1_state_dict_player1': agent1.critic1.state_dict(),
        'critic2_state_dict_player1': agent1.critic2.state_dict(),
        'actor_state_dict_player2': agent2.actor.state_dict(),
        'critic1_state_dict_player2': agent2.critic1.state_dict(),
        'critic2_state_dict_player2': agent2.critic2.state_dict(),
    }, 'sac_agents_connect4.pth')
