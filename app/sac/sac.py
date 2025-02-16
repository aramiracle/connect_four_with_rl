import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from app.environment2 import ConnectFourEnv
from tqdm import tqdm
import numpy as np

# Define the Actor network for SAC
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(6 * 7 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_policy = nn.Linear(64, 7)  # Output for policy logits

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()  # Apply one-hot encoding
        x = x.view(-1, 6 * 7 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy_logits = self.fc_policy(x)
        return policy_logits

# Define the Critic network (Q-function) for SAC - Double Q-Learning, so we need two
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(6 * 7 * 3 + 7, 256) # Input state and action (one-hot encoded)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_value = nn.Linear(64, 1)   # Output Q-value

    def forward(self, state, action):
        state = state.long()
        state_one_hot = F.one_hot(state.to(torch.int64), num_classes=3).float()
        state_flat = state_one_hot.view(-1, 6 * 7 * 3)
        action_one_hot = F.one_hot(action, num_classes=7).float() # Assuming action is an index

        x = torch.cat([state_flat, action_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_value(x)
        return value


# SAC Agent implementation
class SACAgent:
    def __init__(self, env, num_workers=4, gamma=0.99, lr=1e-4, alpha=0.2): # Added alpha, lowered lr
        self.env = env
        self.num_workers = num_workers
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha # Temperature parameter

        self.actor = ActorNet()
        self.critic1 = CriticNet()
        self.critic2 = CriticNet()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def train(self, rollout):
        states, actions, rewards, next_states, dones = rollout

        # Convert actions to torch.long
        actions = actions.long()

        # Critic update
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_action_probs = torch.softmax(next_action_logits, dim=1)
            next_action_dist = Categorical(probs=next_action_probs)
            next_actions = next_action_dist.sample()
            log_prob_next_actions = next_action_dist.log_prob(next_actions).unsqueeze(-1)

            target_q1_values = self.critic1(next_states, next_actions)
            target_q2_values = self.critic2(next_states, next_actions)
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

        actor_loss = (self.alpha * log_prob_actions - q_values).mean() # Maximize Q and Entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def run_episode(self):
        rollout = {
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

            rollout['states'].append(state)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_states'].append(next_state)
            rollout['dones'].append(done)

            state = next_state

            if done:
                break

        return rollout

    def train_async(self, num_episodes=1000):
        for _ in tqdm(range(num_episodes), desc='Training', unit="episode"):
            rollouts = []

            for _ in range(self.num_workers):
                rollout = self.run_episode()
                rollouts.append(rollout)

            for rollout in rollouts:
                self.train(self.unpack_rollout(rollout))

    def unpack_rollout(self, rollout):
        states = torch.stack(rollout['states'])
        actions = torch.tensor(rollout['actions'], dtype=torch.int64)
        rewards = torch.tensor(rollout['rewards'], dtype=torch.float32).unsqueeze(-1) # Ensure reward is [batch_size, 1]
        next_states = torch.stack(rollout['next_states'])
        dones = torch.tensor(rollout['dones'], dtype=torch.float32).unsqueeze(-1)   # Ensure done is [batch_size, 1]
        return states, actions, rewards, next_states, dones

    def select_action(self, state, training=True):
        with torch.no_grad():
            policy_logits = self.actor(state)
            action_probs = torch.softmax(policy_logits, dim=1)

            valid_actions = self.env.get_valid_actions()

            if training:
                # During training, sample an action using multinomial from the valid actions
                valid_action_probs = action_probs[0][valid_actions]
                action_index = torch.multinomial(valid_action_probs, 1).item()
                action = valid_actions[action_index]
            else:
                # During testing, choose the action with the highest probability from the valid actions
                best_valid_action = torch.argmax(action_probs[0, valid_actions]).item()
                action = valid_actions[best_valid_action]

            return action

def agent_vs_agent_train_sac(agents, env, num_episodes=1000):
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training (SAC)", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        episode_rollouts = [
            {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []},
            {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        ]

        while not done:
            for i in range(len(agents)):
                agent = agents[i]
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward

                episode_rollouts[i]['states'].append(state)
                episode_rollouts[i]['actions'].append(action)
                episode_rollouts[i]['rewards'].append(reward)
                episode_rollouts[i]['next_states'].append(next_state)
                episode_rollouts[i]['dones'].append(done)

                state = next_state
                if done:
                    break

        for i in range(len(agents)):
            rollout = episode_rollouts[i]
            if rollout['states']: # Ensure there are states in the rollout
                agents[i].train(agents[i].unpack_rollout(rollout))

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}")

    env.close()


if __name__=='__main__':
    # Example usage:
    env = ConnectFourEnv()
    agent = SACAgent(env, num_workers=4)
    agent.train_async(num_episodes=1000) # Pretrain single agent

    env = ConnectFourEnv()
    agent1 = SACAgent(env, num_workers=4)
    agent2 = SACAgent(env, num_workers=4)
    agent1.actor.load_state_dict(agent.actor.state_dict()) # Share weights from pretrained agent - optional, but can help
    agent1.critic1.load_state_dict(agent.critic1.state_dict())
    agent1.critic2.load_state_dict(agent.critic2.state_dict())
    agent2.actor.load_state_dict(agent.actor.state_dict())
    agent2.critic1.load_state_dict(agent.critic1.state_dict())
    agent2.critic2.load_state_dict(agent.critic2.state_dict())


    agents = [agent1, agent2]
    agent_vs_agent_train_sac(agents, env, num_episodes=10000) # Train agents against each other

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
        'critic2_optimizer_state_dict_player2': agents[1].critic2_optimizer.state_dict(),
    }, 'saved_agents/sac_agents_after_train.pth')