import unittest
from app.environment import ConnectFourEnv
from app.mcts.mcts import Node, MCTSAgent, select, expand, simulate, backpropagate, is_instant_loss

class TestMCTSAgent(unittest.TestCase):
    def test_get_best_action(self):
        env = ConnectFourEnv()
        mcts_agent = MCTSAgent(env, num_simulations=100, depth=2)
        
        # Set up a specific game state for testing
        # For example, let's set up a board with a few moves
        moves = [3, 3, 2, 0, 1, 3, 0]
        for move in moves:
            env.step(move)
        env.render()

        # # Get the root node and perform simulations
        # root_state = env.clone()
        # root_node = Node(root_state)
        # for _ in range(mcts_agent.num_simulations):
        #     node = select(root_node, mcts_agent.depth)
        #     if not node.state.is_terminal():
        #         node = expand(node, mcts_agent.depth)  # Pass depth to expand function
        #         result = simulate(node, mcts_agent.depth)
        #     else:
        #         result = node.state.get_result()

        #     backpropagate(node, result)

        # Get the best action chosen by the agent
        best_action = mcts_agent.get_best_action()
        print("Best Action:", best_action)

        # You can add assertions based on your expectations
        # For example, assert that the chosen action is valid
        valid_actions = env.get_valid_actions()
        self.assertIn(best_action, valid_actions)

        # You may also want to assert other conditions based on your game rules and expectations
        # For example, you can assert that the agent does not choose an instant losing move for the opponent
        self.assertFalse(is_instant_loss(env.clone(), best_action))  # Pass the action to is_instant_loss

if __name__ == "__main__":
    unittest.main()
