import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from app.environment_train import ConnectFourEnv  # Assuming environment_train.py is in the 'app' directory
from tqdm import tqdm
import random
from collections import deque

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize a deque with a fixed maximum length.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Append the new experience to the buffer.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Ensure the sample size doesn't exceed the number of available items.
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Strengthened Actor Network
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout_conv = nn.Dropout2d(0.2)  # dropout for conv features
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_policy = nn.Linear(128, 7)

    def forward(self, x):
        # x: expected shape (batch_size, 6, 7) with integer labels (0,1,2)
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 3, 6, 7)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        policy_logits = self.fc_policy(x)
        return policy_logits

# Strengthened Critic Network (Q-function)
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout_conv = nn.Dropout2d(0.2)
        
        # Fully connected layers incorporate the action input (one-hot of size 7)
        self.fc1 = nn.Linear(128 * 6 * 7 + 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, state, action):
        # state: (batch_size, 6, 7) integer board state
        # action: (batch_size,) integer action to be one-hot encoded
        state = state.long()
        state_one_hot = F.one_hot(state.to(torch.int64), num_classes=3).float()
        state_reshaped = state_one_hot.view(-1, 3, 6, 7)
        
        x = F.relu(self.bn1(self.conv1(state_reshaped)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        
        x = x.view(x.size(0), -1)
        action_one_hot = F.one_hot(action, num_classes=7).float()
        x = torch.cat([x, action_one_hot], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        return value

# SAC Agent implementation
class SACAgent:
    def __init__(self, env, num_workers=4, gamma=0.99, lr=1e-4, alpha=0.2, tau=0.005, buffer_size=100000): # Added tau, buffer_size
        self.env = env
        self.num_workers = num_workers
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha # Temperature parameter
        self.tau = tau # Soft update parameter

        self.actor = ActorNet()
        self.critic1 = CriticNet()
        self.critic2 = CriticNet()
        self.critic1_target = CriticNet() # Target networks
        self.critic2_target = CriticNet()
        self.critic1_target.load_state_dict(self.critic1.state_dict()) # Initialize target networks with current networks
        self.critic2_target.load_state_dict(self.critic2.state_dict())


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size) # Initialize replay buffer


    def soft_update(self, local_model, target_model): # Soft update function
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, rollout=None, batch_size=64): # Modified train function to use replay buffer and batch_size
        if rollout: # For initial pretraining, still accept rollout for single agent training
            states, actions, rewards, next_states, dones = self.unpack_rollout(rollout)
        else: # Sample from replay buffer for agent vs agent training
            if len(self.replay_buffer) < batch_size: # Don't train if not enough samples
                return
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)


        # Critic update (using target networks)
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_action_probs = torch.softmax(next_action_logits, dim=1)
            next_action_dist = Categorical(probs=next_action_probs)
            next_actions = next_action_dist.sample()
            log_prob_next_actions = next_action_dist.log_prob(next_actions).unsqueeze(-1)

            target_q1_values = self.critic1_target(next_states, next_actions) # Use target networks here
            target_q2_values = self.critic2_target(next_states, next_actions) # Use target networks here
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

        # Actor update (same as before)
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

        # Soft updates for target networks after each training step
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


    def run_episode(self): # Modified to add to replay buffer
        rollout = { # Still returns rollout for initial pretraining if needed
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }

        state = self.env.reset()

        while True:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # Store experience in replay buffer
            self.replay_buffer.push(torch.tensor(state).float(), action, reward, torch.tensor(next_state).float(), done) # Store as tensors

            rollout['states'].append(state) # Keep rollout for initial pretraining if needed
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_states'].append(next_state)
            rollout['dones'].append(done)

            state = next_state

            if done:
                break

        return rollout

    def train_async(self, num_episodes=1000, batch_size=64): # Modified to use replay buffer and batch_size in agent vs agent training
        for _ in tqdm(range(num_episodes), desc='Training', unit="episode"):
            rollouts = [] # Keep rollouts for initial pretraining if needed

            for _ in range(self.num_workers):
                rollout = self.run_episode()
                rollouts.append(rollout) # Keep rollouts for initial pretraining if needed

            for rollout in rollouts: # Still train on rollouts for initial pretraining if needed
                self.train(rollout) # Train using rollouts for initial pretraining if needed
                self.train(batch_size=batch_size) # Train from replay buffer for agent vs agent training


    def unpack_rollout(self, rollout):
        states = torch.stack([torch.tensor(s).float() for s in rollout['states']]) # Ensure states are float tensors
        actions = torch.tensor(rollout['actions'], dtype=torch.int64)
        rewards = torch.tensor(rollout['rewards'], dtype=torch.float32).unsqueeze(-1) # Ensure reward is [batch_size, 1]
        next_states = torch.stack([torch.tensor(s).float() for s in rollout['next_states']]) # Ensure next_states are float tensors
        dones = torch.tensor(rollout['dones'], dtype=torch.float32).unsqueeze(-1)   # Ensure done is [batch_size, 1]
        return states, actions, rewards, next_states, dones

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(state).float().unsqueeze(0) # Convert state to tensor and add batch dimension
        with torch.no_grad():
            policy_logits = self.actor(state_tensor)
            action_probs = torch.softmax(policy_logits, dim=1)

            valid_actions = self.env.get_valid_actions()

            if training:
                valid_action_probs = action_probs[0][valid_actions]
                if torch.sum(valid_action_probs) == 0: # Handle case where all valid action probs are zero
                    action = random.choice(valid_actions) # Fallback to random choice
                else:
                    valid_action_probs = valid_action_probs / torch.sum(valid_action_probs) # Normalize in case of numerical issues
                    action_index = torch.multinomial(valid_action_probs, 1).item()
                    action = valid_actions[action_index]
            else:
                best_valid_action = torch.argmax(action_probs[0, valid_actions]).item()
                action = valid_actions[best_valid_action]

            return action
        
class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, training):
        valid_actions = self.env.get_valid_actions()
        return random.choice(valid_actions) if valid_actions else None

def update_elo(rating, opponent_rating, score, k=32):
    """
    Update the ELO rating for a player.
    
    Args:
        rating (float): The player's current rating.
        opponent_rating (float): The opponent's rating.
        score (float): Outcome for the player (1 for win, 0 for loss, 0.5 for draw).
        k (int, optional): The K-factor for the update. Defaults to 32.
    
    Returns:
        float: The updated rating.
    """
    expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
    return rating + k * (score - expected_score)

def run_calibration_game(first_player, second_player, env):
    """
    Run a calibration game between two players in a fixed order.
    
    The first_player makes the first move and the second_player follows.
    Returns the final info dict (expected to contain 'winner' with "Player 1", "Player 2", or "Draw").
    """
    state = env.reset()
    done = False
    while not done:
        # First player's turn
        action = first_player.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
        # Second player's turn
        action = second_player.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
    return info

def calibrate_agent(agent, role, env, k=32):
    """
    Calibrate the given agent's performance against a RandomBot (whose Elo is fixed at 1200).
    
    Args:
        agent: The agent to calibrate.
        role (int): 0 if the agent plays as first player, 1 if as second player.
        env: The environment.
        k (int, optional): The K-factor for the Elo update.
        
    Returns:
        tuple: (calibration_info, old_rating, new_rating)
    """
    random_bot = RandomBot(env)
    if role == 0:
        # Agent plays as first player (RandomBot is second)
        calib_info = run_calibration_game(agent, random_bot, env)
        winner = calib_info.get('winner')
        if winner == "Player 1":
            score = 1
        elif winner == "Draw":
            score = 0.5
        else:
            score = 0
    elif role == 1:
        # Agent plays as second player (RandomBot is first)
        calib_info = run_calibration_game(random_bot, agent, env)
        winner = calib_info.get('winner')
        if winner == "Player 2":
            score = 1
        elif winner == "Draw":
            score = 0.5
        else:
            score = 0
    else:
        raise ValueError("role must be 0 (first player) or 1 (second player)")
    
    old_rating = agent.elo
    agent.elo = update_elo(agent.elo, 1200, score, k)
    return calib_info, old_rating, agent.elo

def agent_vs_agent_train_sac(agents, env, num_episodes=1000, batch_size=64, k=16):
    # Initialize ELO ratings if not already present.
    for agent in agents:
        if not hasattr(agent, 'elo'):
            agent.elo = 1200

    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training (SAC)", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        # Clear replay buffers for each agent (common environment)
        for agent in agents:
            agent.replay_buffer.buffer.clear()
            agent.replay_buffer.position = 0

        # Fixed turn order: agent[0] is first player, agent[1] is second.
        while not done:
            # Agent[0]'s turn (first player)
            action0 = agents[0].select_action(state)
            next_state, reward, done, info = env.step(action0)
            total_rewards[0] += reward

            state_tensor = torch.tensor(state).float()
            next_state_tensor = torch.tensor(next_state).float()
            agents[0].replay_buffer.push(state_tensor, action0, reward, next_state_tensor, done)
            agents[1].replay_buffer.push(state_tensor, action0, reward, next_state_tensor, done)

            state = next_state
            if done:
                break

            # Agent[1]'s turn (second player)
            action1 = agents[1].select_action(state)
            next_state, reward, done, info = env.step(action1)
            total_rewards[1] += reward

            state_tensor = torch.tensor(state).float()
            next_state_tensor = torch.tensor(next_state).float()
            agents[0].replay_buffer.push(state_tensor, action1, reward, next_state_tensor, done)
            agents[1].replay_buffer.push(state_tensor, action1, reward, next_state_tensor, done)

            state = next_state

        # Update ELO ratings based on the game outcome.
        winner = info.get('winner', None)
        if winner is not None:
            if winner == "Player 1":
                score0, score1 = 1, 0
            elif winner == "Draw":
                score0, score1 = 0.5, 0.5
            else:  # Assume "Player 2" win
                score0, score1 = 0, 1
            new_rating_0 = update_elo(agents[0].elo, agents[1].elo, score0, k)
            new_rating_1 = update_elo(agents[1].elo, agents[0].elo, score1, k)
            agents[0].elo, agents[1].elo = new_rating_0, new_rating_1

        # Train each agent from the replay buffer.
        for agent in agents:
            agent.train(batch_size=batch_size)

        tqdm.write(
            f"Episode: {episode}, Winner: {info.get('winner', 'N/A')}, "
            f"ELO P1: {round(agents[0].elo)}, ELO P2: {round(agents[1].elo)}, "
            f"Total Reward P1: {total_rewards[0]:.2f}, Total Reward P2: {total_rewards[1]:.2f}, "
            f"Replay Buffer Size: {len(agents[0].replay_buffer)}"
        )

        # Every 10 episodes, perform calibration for both agents.
        if (episode + 1) % 10 == 0:
            # Calibrate agent[0] as first player.
            calib_info0, old_rating0, new_rating0 = calibrate_agent(agents[0], 0, env, k)
            tqdm.write(
                f"Calibration (Agent 0 as first player): Winner: {calib_info0.get('winner', 'N/A')}, "
                f"ELO updated from {round(old_rating0)} to {round(new_rating0)}"
            )
            # Calibrate agent[1] as second player.
            calib_info1, old_rating1, new_rating1 = calibrate_agent(agents[1], 1, env, k)
            tqdm.write(
                f"Calibration (Agent 1 as second player): Winner: {calib_info1.get('winner', 'N/A')}, "
                f"ELO updated from {round(old_rating1)} to {round(new_rating1)}"
            )

    env.close()

if __name__=='__main__':
    # Example usage:
    env = ConnectFourEnv()
    agent = SACAgent(env, num_workers=4)
    # agent.train_async(num_episodes=1000, batch_size=64) # Pretrain single agent (still uses rollouts)

    env = ConnectFourEnv()
    agent1 = SACAgent(env, num_workers=1) # Reduced num_workers for agent vs agent, replay buffer handles batching
    agent2 = SACAgent(env, num_workers=1)
    agent1.actor.load_state_dict(agent.actor.state_dict()) # Share weights from pretrained agent - optional, but can help
    agent1.critic1.load_state_dict(agent.critic1.state_dict())
    agent1.critic2.load_state_dict(agent.critic2.state_dict())
    agent2.actor.load_state_dict(agent.actor.state_dict())
    agent2.critic1.load_state_dict(agent.critic1.state_dict())
    agent2.critic2.load_state_dict(agent.critic2.state_dict())


    agents = [agent1, agent2]
    agent_vs_agent_train_sac(agents, env, num_episodes=100000, batch_size=128) # Train agents against each other using replay buffer

    # Save the trained model
    torch.save({
        'actor_state_dict_player1': agents[0].actor.state_dict(),
        'critic1_state_dict_player1': agents[0].critic1.state_dict(),
        'critic2_state_dict_player1': agents[0].critic2.state_dict(),
        'actor_optimizer_state_dict_player1': agents[0].actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict_player1': agents[0].critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict_player1': agents[0].critic2_optimizer.state_dict(),

        'actor_state_dict_player2': agents[1].actor.state_dict(),
        'critic1_state_dict_player2': agents[1].critic1.state_dict(),
        'critic2_state_dict_player2': agents[1].critic2.state_dict(),
        'actor_optimizer_state_dict_player2': agents[1].actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict_player2': agents[1].critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict_player2': agents[1].critic2.state_dict(),
    }, 'saved_agents/sac_agents_after_train.pth')