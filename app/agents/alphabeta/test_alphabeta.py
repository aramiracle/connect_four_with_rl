import random
from tqdm import tqdm
from app.agents.alphabeta.alphabeta import AlphaBetaAgent  # Import AlphaBetaAgent
from app.environment_test import ConnectFourEnv # Ensure you are using the correct env file, rename if needed

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, epsilon):
        available_actions = self.env.get_valid_actions() # Use env's method
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if env.current_player == 1:
            if isinstance(player1, AlphaBetaAgent):
                action = player1.select_action(state) # AlphaBeta directly uses state in env
            else:
                action = player1.select_action(state, epsilon=0)
        else:
            if isinstance(player2, AlphaBetaAgent):
                action = player2.select_action(state) # AlphaBeta directly uses state in env
            else:
                action = player2.select_action(state, epsilon=0)
        state, _, terminated, _ = env.step(action) # Get terminated and truncated
    return env.winner

def test_ai_vs_random(env, ai_agent, num_games=1000):
    """Tests an AI agent against a random bot over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0
    ai_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, ai_agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None: # Draw condition in new env might return None or something else, check env.winner after game
            draws += 1

        total_games = ai_wins + random_bot_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')

    return ai_wins, random_bot_wins, draws, ai_win_percentage

def test_random_bot_vs_ai(env, ai_agent, num_games=1000):
    """Tests a random bot against an AI agent over a specified number of games."""
    random_bot_wins = 0
    ai_wins = 0
    draws = 0
    ai_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, random_bot, ai_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            ai_wins += 1
        elif winner is None:
            draws += 1

        total_games = random_bot_wins + ai_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')

    return random_bot_wins, ai_wins, draws, ai_win_percentage


if __name__ == '__main__':
    env = ConnectFourEnv()

    # Initialize AlphaBeta Agent
    ai_agent_alphabeta_player1 = AlphaBetaAgent(env, depth=5, player=1) # Player 1 for AlphaBeta agent
    ai_agent_alphabeta_player2 = AlphaBetaAgent(env, depth=5, player=2) # Player 2 for AlphaBeta agent

    # Test scenarios - AlphaBeta vs Random Bot
    alphabeta_vs_random_results = test_ai_vs_random(env, ai_agent_alphabeta_player1, num_games=1000)
    random_vs_alphabeta_results = test_random_bot_vs_ai(env, ai_agent_alphabeta_player2, num_games=1000)

    # Print results - AlphaBeta vs Random Bot
    print("\n--- AlphaBeta Agent vs Random Bot Results ---")
    print(f"AlphaBeta vs Random Bot: AlphaBeta Wins - {alphabeta_vs_random_results[0]}, Random Bot Wins - {alphabeta_vs_random_results[1]}, Draws - {alphabeta_vs_random_results[2]}, AlphaBeta Win Percentage: {alphabeta_vs_random_results[3]:.2f}%")
    print(f"Random Bot vs AlphaBeta: Random Bot Wins - {random_vs_alphabeta_results[0]}, AlphaBeta Wins - {random_vs_alphabeta_results[1]}, Draws - {random_vs_alphabeta_results[2]}, AlphaBeta Win Percentage: {random_vs_alphabeta_results[3]:.2f}%")

    # Test scenarios - AlphaBeta vs AlphaBeta
    alphabeta_vs_alphabeta_results = test_ai_vs_random(env, ai_agent_alphabeta_player1, num_games=100) # Reduced games for AI vs AI

    # Print results - AlphaBeta vs AlphaBeta
    print("\n--- AlphaBeta Agent vs AlphaBeta Agent Results ---")
    print(f"AlphaBeta Player 1 vs Player 2: Player 1 Wins - {alphabeta_vs_alphabeta_results[0]}, Player 2 Wins - {alphabeta_vs_alphabeta_results[1]}, Draws - {alphabeta_vs_alphabeta_results[2]}, Player 1 Win Percentage: {alphabeta_vs_alphabeta_results[3]:.2f}%")