import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from app.environment_test import ConnectFourEnv
from tqdm import tqdm
import random

# RewardNet Definition (as provided)
class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 256)  # Flatten conv output
        self.fc_reward = nn.Sequential(
            nn.Linear(256, 32),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        state = state.long()
        state_one_hot = F.one_hot(state.to(torch.int64), num_classes=3).float()
        state_reshaped = state_one_hot.view(-1, 3, 6, 7)  # Reshape for CNN
        x = F.relu(self.conv1(state_reshaped))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten conv output
        x = F.relu(self.fc1(x))
        reward = self.fc_reward(x)
        return reward

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer) # Adjust batch size if buffer is smaller
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

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


# SAC Agent implementation
class SACAgent:
    def __init__(self, env, reward_net, num_workers=4, gamma=0.99, lr=1e-4, alpha=0.2, tau=0.005, buffer_size=10000): # Added tau, buffer_size, reward_net
        self.env = env
        self.reward_net = reward_net # Reward Network
        self.reward_net.eval() # Set reward net to eval mode always
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
            if local_param is None:  # Add this check
                print("Warning: local_param is None, skipping update")
                continue
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


    def run_episode(self): # Modified to add to replay buffer and use reward_net
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
            next_state, _, done, _ = self.env.step(action) # Don't use env reward

            # Get reward from RewardNet using NEXT STATE
            reward = self.reward_net(torch.tensor(next_state).float().unsqueeze(0)).item() # Use reward_net on next_state

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

    def load_models(self, save_path):
        checkpoint = torch.load(save_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict_player1'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict_player1'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict_player1'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict_player1'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict_player1'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict_player1'])
        self.critic1_target.load_state_dict(self.critic1.state_dict()) # Ensure target is also loaded/synced
        self.critic2_target.load_state_dict(self.critic2.state_dict())

def agent_vs_agent_train_sac(agents, env, reward_net, num_episodes=1000, batch_size=64): # Modified to use replay buffer and batch_size in agent vs agent training and reward_net
    reward_net.eval() # Ensure reward net is in eval mode
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training (SAC)", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        for i in range(len(agents)): # Clear rollouts - not needed in agent vs agent, use replay buffer
            agents[i].replay_buffer.buffer.clear()
            agents[i].replay_buffer.position = 0


        while not done:
            for i in range(len(agents)):
                agent = agents[i]
                action = agent.select_action(state)
                next_state, _, done, info = env.step(action) # Don't use env reward

                # Get reward from RewardNet using NEXT STATE
                reward = reward_net(torch.tensor(next_state).float().unsqueeze(0)).item() # Use reward_net on next_state
                total_rewards[i] += reward # Sum reward_net rewards - though not directly used in SAC training in this loop

                # Store experience in replay buffer for BOTH agents (common environment)
                agents[0].replay_buffer.push(torch.tensor(state).float(), action, reward, torch.tensor(next_state).float(), done)
                agents[1].replay_buffer.push(torch.tensor(state).float(), action, reward, torch.tensor(next_state).float(), done) # Both agents learn from the same game

                state = next_state
                if done:
                    break

        for i in range(len(agents)):
            agents[i].train(batch_size=batch_size) # Train from replay buffer, no rollout needed in agent vs agent

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]:.2f}, Total Reward Player 2: {total_rewards[1]:.2f}, Replay Buffer Size: {len(agents[0].replay_buffer)}") # Added replay buffer size and RewardNet reward

    env.close()


if __name__=='__main__':
    # Load RewardNet
    reward_net = RewardNet()
    reward_net.load_state_dict(torch.load('saved_reward_network/connect_four_reward_net.pth'))
    reward_net.eval() # Set to eval mode

    # Example usage:
    env = ConnectFourEnv()
    agent = SACAgent(env, reward_net, num_workers=4) # Pass reward_net to SACAgent
    # agent.train_async(num_episodes=1000, batch_size=64) # Pretrain single agent (still uses rollouts)

    # env = ConnectFourEnv()
    agent1 = SACAgent(env, reward_net, num_workers=1) # Reduced num_workers for agent vs agent, replay buffer handles batching, pass reward_net
    agent2 = SACAgent(env, reward_net, num_workers=1) # Pass same reward_net to agent2
    agent1.actor.load_state_dict(agent.actor.state_dict()) # Share weights from pretrained agent - optional, but can help
    agent1.critic1.load_state_dict(agent.critic1.state_dict())
    agent1.critic2.load_state_dict(agent.critic2.state_dict())
    agent2.actor.load_state_dict(agent.actor.state_dict())
    agent2.critic1.load_state_dict(agent.critic1.state_dict())
    agent2.critic2.load_state_dict(agent.critic2.state_dict())


    agents = [agent1, agent2]
    agent_vs_agent_train_sac(agents, env, reward_net, num_episodes=100000, batch_size=128) # Train agents against each other using replay buffer and reward_net

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
    }, 'saved_agents/sac_agents_after_train_rewardnet.pth')