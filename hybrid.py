from dqn import DQNAgent
from mcts import MCTSAgent
from environment import ConnectFourEnv
from collections import namedtuple
import torch
from tqdm import tqdm

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class HybridAgent:
    def __init__(self, env):
        self.env = env
        self.dqn_agent = DQNAgent(self.env)
        self.mcts_agent = MCTSAgent(self.env, dqn_model=self.dqn_agent.model)

    def load_pretrained_dqn_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.dqn_agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.dqn_agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])

    def select_action(self, state, use_mcts=True):
        if use_mcts:
            # Use MCTS to explore and choose an action
            action = self.mcts_agent.select_action(num_simulations=10)
            # Store the result of MCTS simulation in DQN experience buffer
            next_state, reward, done, _ = self.env.step(action)
            self.dqn_agent.buffer.add(Experience(state, action, reward, next_state, done))
            return action
        else:
            # Use DQN to exploit knowledge and choose an action
            epsilon = 0.05  # You can adjust epsilon over time
            return self.dqn_agent.select_action(state, epsilon)

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Train DQN with experiences collected from MCTS
                self.dqn_agent.train_step()

                state = next_state
                if done:
                    break

            # Periodically update the DQN model used within MCTS
            if episode % self.dqn_agent.target_update_frequency == 0:
                self.mcts_agent.dqn_model.load_state_dict(self.dqn_agent.model.state_dict())


if __name__=='__main__':
    env = ConnectFourEnv()
    hybrid_agent = HybridAgent(env)

    # Load the pre-trained DQN model
    hybrid_agent.load_pretrained_dqn_model('saved_agents/dqn_agent_after_training.pth')


    # Train the DQN agent
    num_episodes = 10000
    hybrid_agent.train(num_episodes=num_episodes)

    # Save the DQN agent's state after training
    torch.save({
    'model_state_dict': hybrid_agent.dqn_agent.model.state_dict(),
    'target_model_state_dict': hybrid_agent.dqn_agent.target_model.state_dict(),
    'optimizer_state_dict': hybrid_agent.dqn_agent.optimizer.state_dict(),
    }, 'saved_agents/hybrid_agent_after_training.pth')
