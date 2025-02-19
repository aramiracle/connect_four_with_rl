import torch
import random
from tqdm import tqdm
from app.hybrid.hybrid_sac import HybridSACAgent
from app.environment_test import ConnectFourEnv
from app.alphabeta.alphabeta import AlphaBetaAgent

def simulate_game(env, player1, player2, epsilon=0):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            if isinstance(player1, AlphaBetaAgent):
                action = player1.select_action(state) # AlphaBetaAgent doesn't use epsilon
            else:
                action = player1.select_action(state, epsilon)
        else:
            if isinstance(player2, AlphaBetaAgent):
                action = player2.select_action(state) # AlphaBetaAgent doesn't use epsilon
            else:
                action = player2.select_action(state, epsilon)
        state, _, done, _ = env.step(action)
    return env.winner

def test_hybrid_vs_alphabeta(env, hybrid_agent, alphabeta_agent, num_games=100): # Reduced num_games for AlphaBeta vs Hybrid as it's slower
    """Tests a Hybrid Agent against an AlphaBeta Agent over a specified number of games."""
    hybrid_wins = 0
    alphabeta_wins = 0
    draws = 0
    hybrid_win_percentage = 0.0

    pbar = tqdm(range(num_games), desc=f'Hybrid vs AlphaBeta (Hybrid Win%: {hybrid_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, hybrid_agent, alphabeta_agent)

        if winner == 1:
            hybrid_wins += 1
        elif winner == 2:
            alphabeta_wins += 1
        elif winner is None:
            draws += 1

        total_games = hybrid_wins + alphabeta_wins + draws
        hybrid_win_percentage = (hybrid_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'Hybrid vs AlphaBeta (Hybrid Win%: {hybrid_win_percentage:.2f}%)')

    return hybrid_wins, alphabeta_wins, draws, hybrid_win_percentage

def test_alphabeta_vs_hybrid(env, alphabeta_agent, hybrid_agent, num_games=100): # Reduced num_games for AlphaBeta vs Hybrid as it's slower
    """Tests an AlphaBeta Agent against a Hybrid Agent over a specified number of games."""
    alphabeta_wins = 0
    hybrid_wins = 0
    draws = 0
    alphabeta_win_percentage = 0.0 # Changed to alphabeta_win_percentage

    pbar = tqdm(range(num_games), desc=f'AlphaBeta vs Hybrid (AlphaBeta Win%: {alphabeta_win_percentage:.2f}%)') # Changed description
    for _ in pbar:
        winner = simulate_game(env, alphabeta_agent, hybrid_agent)

        if winner == 1:
            alphabeta_wins += 1
        elif winner == 2:
            hybrid_wins += 1
        elif winner is None:
            draws += 1

        total_games = alphabeta_wins + hybrid_wins + draws
        alphabeta_win_percentage = (alphabeta_wins + draws / 2) / total_games * 100 if total_games > 0 else 0 # Changed to alphabeta_win_percentage
        pbar.set_description(f'AlphaBeta vs Hybrid (AlphaBeta Win%: {alphabeta_win_percentage:.2f}%)') # Changed description

    return alphabeta_wins, hybrid_wins, draws, alphabeta_win_percentage # Changed return value

if __name__ == '__main__':
    # Create environment
    env = ConnectFourEnv()

    # Load AI agents
    ai_agent_player1 = HybridSACAgent(env, player_piece=1)  # Use DQN class directly
    checkpoint_player1 = torch.load('saved_agents/sac_agents_after_train.pth')
    ai_agent_player1.actor.load_state_dict(checkpoint_player1['actor_state_dict_player1'])
    ai_agent_player1.critic1.load_state_dict(checkpoint_player1['critic1_state_dict_player1'])
    ai_agent_player1.critic2.load_state_dict(checkpoint_player1['critic2_state_dict_player1'])

    ai_agent_player2 = HybridSACAgent(env, player_piece=2)  # Use DQN class directly
    checkpoint_player2 = torch.load('saved_agents/sac_agents_after_train.pth')
    ai_agent_player2.actor.load_state_dict(checkpoint_player2['actor_state_dict_player2'])
    ai_agent_player2.critic1.load_state_dict(checkpoint_player2['critic1_state_dict_player2'])
    ai_agent_player2.critic2.load_state_dict(checkpoint_player2['critic2_state_dict_player2'])

    # Create AlphaBeta Agents
    alphabeta_agent_player1 = AlphaBetaAgent(env, depth=5, player=1) # Depth 4 for reasonable time, player 1
    alphabeta_agent_player2 = AlphaBetaAgent(env, depth=5, player=2) # Depth 4 for reasonable time, player 2


    # Test scenarios
    hybrid_vs_alphabeta_results = test_hybrid_vs_alphabeta(env, ai_agent_player1, alphabeta_agent_player2, num_games=100) # Test Hybrid vs AlphaBeta
    alphabeta_vs_hybrid_results = test_alphabeta_vs_hybrid(env, alphabeta_agent_player1, ai_agent_player2, num_games=100) # Test AlphaBeta vs Hybrid


    # Print results
    print(f"Hybrid vs AlphaBeta Results: Hybrid Wins - {hybrid_vs_alphabeta_results[0]}, AlphaBeta Wins - {hybrid_vs_alphabeta_results[1]}, Draws - {hybrid_vs_alphabeta_results[2]}, Hybrid Win Percentage: {hybrid_vs_alphabeta_results[3]:.2f}%")
    print(f"AlphaBeta vs Hybrid Results: AlphaBeta Wins - {alphabeta_vs_hybrid_results[0]}, Hybrid Wins - {alphabeta_vs_hybrid_results[1]}, Draws - {alphabeta_vs_hybrid_results[2]}, AlphaBeta Win Percentage: {alphabeta_vs_hybrid_results[3]:.2f}%")