import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from app.environment_train import ConnectFourEnv
from tqdm import tqdm
import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer (reuse from sac.py or hybrid.py - using hybrid.py's ExperienceReplayBuffer for consistency with HybridAgent)
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

# is_instant_win and is_instant_loss (reuse from hybrid.py)
def is_instant_win(board, player_piece):
    piece = int(player_piece) # Ensure piece is int for comparison
    rows, cols = board.shape
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3] == piece:
                return True
    # Check vertical
    for r in range(rows - 3):
        for c in range(cols):
            if board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c] == piece:
                return True
    # Check diagonals (top-left to bottom-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3] == piece:
                return True
    # Check diagonals (top-right to bottom-left)
    for r in range(rows - 3):
        for c in range(3, cols):
            if board[r, c] == board[r + 1, c - 1] == board[r + 2, c - 2] == board[r + 3, c - 3] == piece:
                return True
    return False

def is_instant_loss(board, player_piece):
    opponent_piece = 3 - int(player_piece) # Opponent piece is 3 - player_piece (if player is 1, opponent is 2, if player is 2, opponent is 1)
    return is_instant_win(board, opponent_piece) # Loss for current player is win for opponent

# Actor and Critic Networks (reuse from sac.py - keeping them same for fair comparison, can adjust if needed)
# Define the Actor network with CNN
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # Input channels: 3 (one-hot encoded), Output channels: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 128) # Flatten conv output
        self.fc_policy = nn.Linear(128, 7)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 3, 6, 7) # Reshape to (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        return policy_logits

# Define the Critic network (Q-function) with CNN
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7 + 7, 128) # +7 for action input
        self.fc_value = nn.Linear(128, 1)

    def forward(self, state, action):
        state = state.long()
        state_one_hot = F.one_hot(state.to(torch.int64), num_classes=3).float()
        state_reshaped = state_one_hot.view(-1, 3, 6, 7) # Reshape for CNN
        x = F.relu(self.conv1(state_reshaped))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten conv output
        action_one_hot = F.one_hot(action, num_classes=7).float()
        x = torch.cat([x, action_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        return value

# Hybrid SAC Agent - Combining SAC with Instant Win/Loss Logic
class HybridSACAgent:
    def __init__(self, env, player_piece, num_workers=1, gamma=0.99, lr=1e-4, alpha=0.2, tau=0.005, buffer_size=10000):
        self.env = env
        self.player_piece = player_piece # Store player piece
        self.num_workers = num_workers
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha # Temperature parameter
        self.tau = tau # Soft update parameter
        self.buffer_size = buffer_size

        self.actor = ActorNet().to(device)
        self.critic1 = CriticNet().to(device)
        self.critic2 = CriticNet().to(device)
        self.critic1_target = CriticNet().to(device)
        self.critic2_target = CriticNet().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)

        # Critic update
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_action_probs = torch.softmax(next_action_logits, dim=1)
            next_action_dist = Categorical(probs=next_action_probs)
            next_actions = next_action_dist.sample()
            log_prob_next_actions = next_action_dist.log_prob(next_actions).unsqueeze(-1)

            target_q1_values = self.critic1_target(next_states, next_actions)
            target_q2_values = self.critic2_target(next_states, next_actions)
            target_q_values = torch.min(target_q1_values, target_q2_values)
            target_q = rewards + (1 - dones) * self.gamma * (target_q_values - self.alpha * log_prob_next_actions)

        current_q1_values = self.critic1(states, actions)
        current_q2_values = self.critic2(states, actions)

        critic1_loss = self.mse_loss(current_q1_values, target_q)
        critic2_loss = self.mse_loss(current_q2_values, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        actor_policy_logits = self.actor(states)
        actor_policy_probs = torch.softmax(actor_policy_logits, dim=1)
        actor_policy_dist = Categorical(probs=actor_policy_probs)
        sampled_actions = actor_policy_dist.sample()
        log_prob_actions = actor_policy_dist.log_prob(sampled_actions).unsqueeze(-1)

        q1_values = self.critic1(states, sampled_actions)
        q2_values = self.critic2(states, sampled_actions)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = (self.alpha * log_prob_actions - q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


    def select_action(self, state, training=True):
        valid_actions = self.env.get_valid_actions()

        # Check for instant win moves
        instant_win_actions = [action for action in valid_actions if self.is_instant_win(self.env, action)]
        if instant_win_actions:
            return random.choice(instant_win_actions)

        # Filter out instant loss moves
        filtered_actions = []
        for action in valid_actions:
            if not self.is_instant_loss(self.env, action):
                temp_env = self.env.clone()
                temp_env.step(action)
                opponent_can_instant_win = False
                for opponent_action in temp_env.get_valid_actions():
                    if self.is_instant_win(temp_env, opponent_action, 3 - self.player_piece):
                        opponent_can_instant_win = True
                        break
                if not opponent_can_instant_win:
                    filtered_actions.append(action)

        if filtered_actions:
            valid_actions_sac = filtered_actions # Use filtered actions for SAC policy
        else:
            valid_actions_sac = valid_actions # Fallback to all valid actions if no filtered actions

        state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            self.actor.eval() # Set actor to eval mode for inference
            policy_logits = self.actor(state_tensor).cpu() # Move logits to CPU for action selection
            action_probs = torch.softmax(policy_logits, dim=1)
            self.actor.train() # Set actor back to train mode

            valid_action_probs = action_probs[0][valid_actions_sac]

            if training:
                if torch.sum(valid_action_probs) == 0:
                    action = random.choice(valid_actions_sac)
                else:
                    valid_action_probs = valid_action_probs / torch.sum(valid_action_probs)
                    action_index = torch.multinomial(valid_action_probs, 1).item()
                    action = valid_actions_sac[action_index]
            else:
                best_valid_action = torch.argmax(action_probs[0, valid_actions_sac]).item()
                action = valid_actions_sac[best_valid_action]
            return action


    def is_instant_win(self, env, action, player_piece=None):
        if player_piece is None:
            player_piece = self.player_piece
        next_env = env.clone()
        next_env.step(action)
        board_np = next_env.board.cpu().numpy()
        return is_instant_win(board_np, player_piece)

    def is_instant_loss(self, env, action):
        next_env = env.clone()
        next_env.step(action)
        board_np = next_env.board.cpu().numpy()
        return is_instant_loss(board_np, self.player_piece)

def agent_vs_agent_train_hybrid_sac(agents, env, num_episodes=10000, batch_size=128):
    reward_history_p1 = []
    reward_history_p2 = []

    for episode in tqdm(range(num_episodes), desc="Hybrid SAC Agent vs Agent Training", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False
        player_turn = 1
        agent_index = 0

        for agent in agents: # Clear buffer at start of episode for clean learning per episode - optional, can remove
            agent.replay_buffer.buffer.clear()
            agent.replay_buffer.position = 0

        while not done:
            agent = agents[agent_index]
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_rewards[agent_index] += reward

            # Store experience in replay buffer for BOTH agents
            agents[0].replay_buffer.add(Experience(torch.tensor(state).float(), action, reward, torch.tensor(next_state).float(), done))
            agents[1].replay_buffer.add(Experience(torch.tensor(state).float(), action, reward, torch.tensor(next_state).float(), done))

            state = next_state
            if done:
                break

            player_turn = 3 - player_turn
            agent_index = 1 - agent_index

        # Train agents
        for agent in agents:
            agent.train_step(batch_size=batch_size)

        reward_history_p1.append(total_rewards[0])
        reward_history_p2.append(total_rewards[1])

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}, Replay Buffer Size: {len(agents[0].replay_buffer)}")

        if episode % 1000 == 0 and episode > 0:
            avg_reward_p1 = sum(reward_history_p1[-1000:]) / 1000
            avg_reward_p2 = sum(reward_history_p2[-1000:]) / 1000
            tqdm.write(f"--- Episode {episode} Report ---")
            tqdm.write(f"Average Reward over last 1000 episodes - Player 1: {avg_reward_p1:.2f}, Player 2: {avg_reward_p2:.2f}")
            tqdm.write(f"------------------------------")

    env.close()


if __name__ == '__main__':
    env = ConnectFourEnv()

    # Initialize Hybrid SAC Agents, passing player_piece
    hybrid_sac_agents = [HybridSACAgent(env, player_piece=1), HybridSACAgent(env, player_piece=2)]

    # Train Hybrid SAC Agents against each other
    agent_vs_agent_train_hybrid_sac(hybrid_sac_agents, env, num_episodes=30000, batch_size=128)

    # Save the trained Hybrid SAC agents
    torch.save({
        'actor_state_dict_player1': hybrid_sac_agents[0].actor.state_dict(),
        'critic1_state_dict_player1': hybrid_sac_agents[0].critic1.state_dict(),
        'critic2_state_dict_player1': hybrid_sac_agents[0].critic2.state_dict(),
        'actor_optimizer_state_dict_player1': hybrid_sac_agents[0].actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict_player1': hybrid_sac_agents[0].critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict_player1': hybrid_sac_agents[0].critic2_optimizer.state_dict(),

        'actor_state_dict_player2': hybrid_sac_agents[1].actor.state_dict(),
        'critic1_state_dict_player2': hybrid_sac_agents[1].critic1.state_dict(),
        'critic2_state_dict_player2': hybrid_sac_agents[1].critic2.state_dict(),
        'actor_optimizer_state_dict_player2': hybrid_sac_agents[1].actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict_player2': hybrid_sac_agents[1].critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict_player2': hybrid_sac_agents[1].critic2.state_dict(),
    }, 'saved_agents/hybrid_sac_agents_after_train.pth')