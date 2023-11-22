import torch
import numpy as np
from app.environment import ConnectFourEnv
from tqdm import tqdm
from app.ddqn.ddqn import DDQNAgent, ExperienceReplayBuffer, Experience
from app.mcts.mcts import MCTSAgent


# Hybrid Agent combining MCTS and DDQN
class HybridAgent:
    def __init__(self, env, buffer_capacity=1000000, num_simulations=20, depth=10):
        self.env = env
        self.num_simulations = num_simulations
        self.depth = depth
        self.ddqn_agent = DDQNAgent(env)
        self.mcts_agent = MCTSAgent(env, num_simulations=num_simulations, depth=depth)
        self.buffer = ExperienceReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state, epsilon):
        # Use MCTS to select the best action
        if np.random.rand() < epsilon:
            return self.ddqn_agent.select_action(state, epsilon=0)
        else:
            self.mcts_agent.env.render()
            action = self.mcts_agent.get_best_action()
            print(action)
            return action
        
def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=0, epsilon_final=0.01, epsilon_decay=0.9999):
    epsilon = epsilon_start

    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                total_rewards[i] += reward
                agents[i].buffer.add(Experience(state, action, reward, next_state, done))
                state = next_state
                if done:
                    if env.winner == 1:
                        total_rewards[1] = -total_rewards[0]
                    elif env.winner == 2:
                        total_rewards[0] = -total_rewards[1]
                    break

        # Batch processing of experiences for each agent
        for agent in agents:
            agent.ddqn_agent.train_step()

        tqdm.write(f"Episode: {episode}, Winner: {env.winner}, Total Reward Player 1: {total_rewards[0]:.4f}, Total Reward Player 2: {total_rewards[1]:.4f}, Epsilon: {epsilon:.2f}")

        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Load DDQN agents
def load_ddqn_agent(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    agent.ddqn_agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
    agent.ddqn_agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
    agent.ddqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict_player1'])
    return agent

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players
    hybrid_agents = [HybridAgent(env), HybridAgent(env)]

    # Load DDQN agents for player 1 and player 2
    hybrid_agents[0] = load_ddqn_agent(hybrid_agents[0], 'saved_agents/ddqnd_agents_after_train.pth')
    hybrid_agents[1] = load_ddqn_agent(hybrid_agents[1], 'saved_agents/ddqnd_agents_after_train.pth')

    # Agent vs Agent Training
    agent_vs_agent_train(hybrid_agents, env, num_episodes=3000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': hybrid_agents[0].ddqn_agent.model.state_dict(),
        'target_model_state_dict_player1': hybrid_agents[0].ddqn_agent.target_model.state_dict(),
        'optimizer_state_dict_player1': hybrid_agents[0].ddqn_agent.optimizer.state_dict(),
        'model_state_dict_player2': hybrid_agents[1].ddqn_agent.model.state_dict(),
        'target_model_state_dict_player2': hybrid_agents[1].ddqn_agent.target_model.state_dict(),
        'optimizer_state_dict_player2': hybrid_agents[1].ddqn_agent.optimizer.state_dict(),
    }, 'saved_agents/hybrid_agents_after_train.pth')