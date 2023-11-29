from tqdm import tqdm
from app.environment import ConnectFourEnv
from app.mcts.mcts import MCTSAgent
import random

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        available_actions = [col for col in range(self.env.action_space.n) if self.env.board[0][col] == 0]
        return random.choice(available_actions) if available_actions else None


def simulate_game(env, agent1, agent2):
    state = env.reset()
    done = False
    while not done:
        action = agent1.select_action() if env.current_player == 1 else agent2.select_action()
        state, _, done, _ = env.step(action)
    return env.winner

def test_agents(env, agent1, agent2, num_games=100):
    wins1, wins2, draws = 0, 0, 0
    for _ in tqdm(range(num_games)):
        winner = simulate_game(env, agent1, agent2)
        if winner == 1:
            wins1 += 1
        elif winner == 2:
            wins2 += 1
        elif winner is None:
            draws += 1
    return wins1, wins2, draws

if __name__ == '__main__':
    env = ConnectFourEnv()

    mcst_agent1 = MCTSAgent(env, num_simulations=100, depth=2)
    mcst_agent2 = MCTSAgent(env, num_simulations=100, depth=2)
    random_bot = RandomBot(env)

    mcst_vs_random_results = test_agents(env, mcst_agent1, random_bot)
    random_vs_mcst_results = test_agents(env, random_bot, mcst_agent2)
    mcst_vs_mcst_results = test_agents(env, mcst_agent1, mcst_agent2)

    print(f"MCTS vs Random Bot: MCTS Wins - {mcst_vs_random_results[0]}, Random Bot Wins - {mcst_vs_random_results[1]}, Draws - {mcst_vs_random_results[2]}")
    print(f"Random Bot vs MCTS: Random Bot Wins - {random_vs_mcst_results[0]}, MCTS Wins - {random_vs_mcst_results[1]}, Draws - {random_vs_mcst_results[2]}")
    print(f"MCTS vs MCTS: Player 1 Wins - {mcst_vs_mcst_results[0]}, Player 2 Wins - {mcst_vs_mcst_results[1]}, Draws - {mcst_vs_mcst_results[2]}")
