from app.dqn.dqn import DQNAgent
from app.dqn.mcts import MCTSAgent
from app.environment import ConnectFourEnv
import torch
from tqdm import tqdm

class HybridAgent:
    def __init__(self, env, num_simulations=100, depth=2):
        # Initialize a HybridAgent with DQN and MCTS components
        self.env = env
        self.dqn_agent_player1 = DQNAgent(self.env)
        self.dqn_agent_player2 = DQNAgent(self.env)  # Create a separate instance for player2
        self.mcts_agent = MCTSAgent(self.env, num_simulations=num_simulations, depth=depth)

    def load_pretrained_dqn_model(self, filepath):
        # Load pre-trained DQN models for both players
        checkpoint = torch.load(filepath)
        self.dqn_agent_player1.model.load_state_dict(checkpoint['model_state_dict_player1'])
        self.dqn_agent_player1.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
        self.dqn_agent_player1.optimizer.load_state_dict(checkpoint['optimizer_state_dict_player1'])

        # Load player2's models
        self.dqn_agent_player2.model.load_state_dict(checkpoint['model_state_dict_player2'])
        self.dqn_agent_player2.target_model.load_state_dict(checkpoint['target_model_state_dict_player2'])
        self.dqn_agent_player2.optimizer.load_state_dict(checkpoint['optimizer_state_dict_player2'])

    def select_action(self, state, player, epsilon=0, use_mcts=True):
        # Select an action using either MCTS or DQN, based on the specified conditions
        if use_mcts:
            action = self.mcts_agent.get_best_action()
            if action is not None:
                return action
            else:
                # Handle the case where MCTS fails to find a valid action
                print("MCTS failed to find a valid action. Using DQN instead.")
                use_mcts = False

        # Use DQN to exploit knowledge and choose an action
        epsilon = 0
        if player == 1:
            return self.dqn_agent_player1.select_action(state, epsilon)
        else:
            return self.dqn_agent_player2.select_action(state, epsilon)

    def train_step(self):
        # Perform a training step for both DQN agents
        self.dqn_agent_player1.train_step()
        self.dqn_agent_player2.train_step()

def agent_vs_agent_train(agents, env, num_episodes=1000):
    # Train two HybridAgents in an adversarial environment
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        states = [env.reset(), env.reset()]
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state=states[i], player=i+1, use_mcts=True)
                next_state, reward, done, _ = env.step(action)
                total_rewards[i] += reward
                states[i] = next_state

                if done:
                    total_rewards[1 - i] = -total_rewards[i]
                    break

        agents[0].train_step()
        agents[1].train_step()

        tqdm.write(
            f"Episode: {episode}, Total Reward Player 1: {total_rewards[0]:.4f}, Total Reward Player 2: {total_rewards[1]:.4f}"
        )

    env.close()

if __name__ == "__main__":
    env = ConnectFourEnv()

    # Hybrid Agents
    hybrid_agents = [HybridAgent(env, num_simulations=1000, depth=2) for _ in range(2)]

    # Agent vs Agent Training
    agent_vs_agent_train(hybrid_agents, env, num_episodes=1)

    # Save the trained hybrid agents
    torch.save(
        {
            "model_state_dict_player1": hybrid_agents[0].dqn_agent_player1.model.state_dict(),
            "target_model_state_dict_player1": hybrid_agents[0].dqn_agent_player1.target_model.state_dict(),
            "optimizer_state_dict_player1": hybrid_agents[0].dqn_agent_player1.optimizer.state_dict(),
            "model_state_dict_player2": hybrid_agents[1].dqn_agent_player2.model.state_dict(),
            "target_model_state_dict_player2": hybrid_agents[1].dqn_agent_player2.target_model.state_dict(),
            "optimizer_state_dict_player2": hybrid_agents[1].dqn_agent_player2.optimizer.state_dict(),
        },
        "saved_agents/hybrid_agents_after_train.pth",
    )
