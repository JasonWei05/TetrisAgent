"""ES Look-Ahead Tetris Agent

Script to run the Evolution Strategy (ES) with lookahead capability,
examining 1-4 tetrominoes in the queue as specified by command line argument.
"""
import gymnasium as gym
import sys
import argparse
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../Tetris-Gymnasium')

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

class Next:
    def __init__(self, env):
        self.env = env.env
    
    def collision(self, tetromino, x: int, y: int, board) -> bool:
        """Check if the tetromino collides with the board at the given position."""
        slices = self.get_tetromino_slices(tetromino, x, y)
        board_subsection = board.copy()[slices]
        return np.any(board_subsection[tetromino.matrix > 0] > 0)
    
    def get_tetromino_slices(self, tetromino, x: int, y: int):
        """Get the slices of the active tetromino on the board."""
        tetromino_height, tetromino_width = tetromino.matrix.shape
        return (slice(y, y + tetromino_height), slice(x, x + tetromino_width))
    
    def collision_with_frame(self, tetromino, x: int, y: int, board) -> bool:
        """Check if the tetromino collides with the frame."""
        slices = self.env.unwrapped.get_tetromino_slices(tetromino, x, y)
        board_subsection = board.copy()[slices]
        return np.any(board_subsection[tetromino.matrix > 0] == 1)
    
    def project_tetromino(self, tetromino=None, x: int=None, y: int=None, board=None) -> np.ndarray:
        """Project the active tetromino on the board."""
        projection = board.copy()
        if self.collision(tetromino, x, y, projection):
            return projection
        slices = self.get_tetromino_slices(tetromino, x, y)
        projection[slices] += tetromino.matrix
        return projection

    def observation(self, env, observation, tetrimino):
        """Generate all legal placements for the current tetromino."""
        board_obs = observation.copy()
        self.legal_actions_mask = np.ones(env.action_space.n)
        grouped_board_obs = []
        t = tetrimino
        for x in range(self.env.unwrapped.width):
            # Adjust x position based on padding
            x_pos = self.env.unwrapped.padding + x
            for r in range(4):
                y = 0
                if r > 0:
                    t = self.env.unwrapped.rotate(t)
                while not self.collision(t, x_pos, y + 1, board_obs):
                    y += 1
                if self.collision_with_frame(t, x_pos, y, board_obs) or self.collision(t, x_pos, y, board_obs):
                    # illegal placement; mark with an all-ones board
                    grouped_board_obs.append(np.ones_like(board_obs))
                else:
                    # legal placement; project the tetromino onto the board
                    grouped_board_obs.append(self.project_tetromino(t, x_pos, y, board_obs))
            t = self.env.unwrapped.rotate(t)  # Reset rotation (has been rotated 3 times)
        grouped_board_obs = np.array(grouped_board_obs)
        return grouped_board_obs

class FeatureVector:
    """Generates a feature vector from the board state."""
    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483

    def __init__(self, report_height=True, report_max_height=True, report_holes=True, report_bumpiness=True):
        self.report_height = report_height
        self.report_max_height = report_max_height
        self.report_holes = report_holes
        self.report_bumpiness = report_bumpiness

    def calc_height(self, board):
        heights = board.shape[0] - np.argmax(board != 0, axis=0)
        heights = np.where(np.all(board == 0, axis=0), 0, heights)
        return heights

    def calc_max_height(self, board):
        return np.max(self.calc_height(board))

    def calc_bumpiness(self, board):
        heights = self.calc_height(board)
        return np.sum(np.abs(np.diff(heights)))

    def calc_holes(self, board):
        filled = board != 0
        cumsum = np.cumsum(filled, axis=0)
        return np.sum((board == 0) & (cumsum > 0))

    def complete_lines(self, grid):
        if np.all(grid == 1):
            return 0
        return sum(1 for row in grid if all(cell != 0 for cell in row))

    def observation(self, observation, padding=4):
        board_obs = observation[0:-padding, padding:-padding]
        features = []
        if self.report_height or self.report_max_height:
            height_vector = self.calc_height(board_obs)
            if self.report_height:
                features += list(height_vector)
            if self.report_max_height:
                features.append(np.max(height_vector))
        if self.report_holes:
            features.append(self.calc_holes(board_obs))
        if self.report_bumpiness:
            features.append(self.calc_bumpiness(board_obs))
        return np.array(features, dtype=np.uint8)

    def value(self, observation, padding=4, prt=False):
        board_obs = observation[0:-padding, padding:-padding]
        agg_height = sum(self.calc_height(board_obs))
        complete_lines = self.complete_lines(board_obs)
        holes = self.calc_holes(board_obs)
        bumpiness = self.calc_bumpiness(board_obs)
        if prt:
            print(f"Aggregate Height: {agg_height}")
            print(f"Complete Lines: {complete_lines}")
            print(f"Holes: {holes}")
            print(f"Bumpiness: {bumpiness}")
        return self.a * agg_height + self.b * complete_lines + self.c * holes + self.d * bumpiness

def valid_indices(obs):
    """Return indices of observations that are legal placements."""
    indices = []
    for i in range(len(obs)):
        if not (np.all(obs[i] == 1) or np.all(obs[i] == 0)):
            indices.append(i)
    return indices

class BeamSearch:
    """
    Implements beam search for selecting the best move in Tetris without taking random actions after the first step.
    
    The search expands nodes up to a fixed lookahead depth (num_look_ahead), maintaining a beam of the best nodes
    at each level based on FeatureVector.value. Each node stores the first move taken
    from the root. After reaching the desired depth, the best node is chosen and its stored first move is returned.
    """
    def __init__(self, env, beam_width=10, lookahead_steps=2):
        """
        Args:
            env: The Tetris environment.
            beam_width (int): Number of nodes to keep at each level.
            lookahead_steps (int): Number of steps to look ahead (num_look_ahead).
        """
        self.env = env
        self.beam_width = beam_width
        self.lookahead_steps = lookahead_steps
        self.feature_vector = FeatureVector()

    def best_action(self, board, tetromino, upcoming):
        """
        Perform beam search starting from the current board state and active tetromino.
        
        Args:
            board (np.ndarray): Current board state.
            tetromino: The active tetromino.
            upcoming (list): List of upcoming tetromino indices (its length should match lookahead_steps).
        
        Returns:
            int: The initial move (index) that leads to the best outcome after the lookahead.
        """
        nxt = Next(self.env)
        root = {
            'board': board,
            'tetromino': tetromino,
            'first_move': None,
            'parent': None,
            'depth': 0,
            'evaluation': None
        }
        
        # Initialize the beam with the root node.
        beam = [root]
        
        # Expand the beam for the specified number of lookahead steps.
        for step in range(self.lookahead_steps):
            new_beam = []
            for node in beam:
                # If there is no tetromino to place, treat this node as terminal.
                if node['tetromino'] is None:
                    new_beam.append(node)
                    continue
                moves = nxt.observation(self.env, node['board'], node['tetromino'])
                valid_moves = valid_indices(moves)
                
                # If no legal moves exist, carry forward this node.
                if not valid_moves:
                    new_beam.append(node)
                    continue
                
                for move in valid_moves:
                    new_board = moves[move]
                    # Use the upcoming list for the next tetromino.
                    if step < len(upcoming):
                        next_tet = self.env.env.unwrapped.TETROMINOES[upcoming[step]]
                    else:
                        next_tet = None  # Should not happen if upcoming is of proper length.
                    
                    # For nodes directly expanded from the root, store the move as the first_move.
                    first_move = move if node['depth'] == 0 else node['first_move']
                    eval_value = self.feature_vector.value(new_board)
                    new_node = {
                        'board': new_board,
                        'tetromino': next_tet,
                        'first_move': first_move,
                        'parent': node,
                        'depth': node['depth'] + 1,
                        'evaluation': eval_value
                    }
                    new_beam.append(new_node)
            
            if not new_beam:
                break
            
            # Retain only the top beam_width nodes based on evaluation.
            beam = sorted(new_beam, key=lambda n: n['evaluation'], reverse=True)[:self.beam_width]
        
        # After expanding for the lookahead steps, select the node with the highest evaluation.
        best_node = max(beam, key=lambda n: n['evaluation'])
        # print("Beam Search best final evaluation:", best_node['evaluation'])
        
        # The best node's stored first_move is the move from the current state that led to the best outcome.
        return best_node['first_move']

class MCTSNode:
    """A node in the Monte Carlo Tree Search representing a game state in Tetris."""
    
    def __init__(self, env, board, tetromino, parent=None, move=None, queue_index=0, depth=0, known_queue=None):
        self.env = env
        self.board = board
        self.tetromino = tetromino
        self.parent = parent
        self.move = move  # The index (in the list from Next.observation) that led to this state
        self.children = {}  # Maps move indices to child nodes
        self.visits = 0
        self.total_value = 0.0
        self.queue_index = queue_index  # How many tetrominoes from known_queue have been used so far
        self.depth = depth
        self.known_queue = known_queue if known_queue is not None else []
        
        # Get all projected moves for the current board and tetromino.
        nxt = Next(env)
        self.all_moves = nxt.observation(env, board, tetromino)
        self.untried_moves = valid_indices(self.all_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        """
        Create a new child node by trying one of the untried moves.
        
        Returns:
            MCTSNode: The newly created child node.
        """
        move = self.untried_moves.pop()
        new_board = self.all_moves[move]
        
        # Determine the next tetromino:
        if self.queue_index < len(self.known_queue):
            next_tet = self.env.env.unwrapped.TETROMINOES[self.known_queue[self.queue_index]]
            new_queue_index = self.queue_index + 1
        else:
            next_tet = random.choice(self.env.env.unwrapped.TETROMINOES)
            new_queue_index = self.queue_index
        
        child_node = MCTSNode(
            self.env,
            new_board,
            next_tet,
            parent=self,
            move=move,
            queue_index=new_queue_index,
            depth=self.depth + 1,
            known_queue=self.known_queue
        )
        
        self.children[move] = child_node
        return child_node


class MCTS:
    def __init__(self, env, known_queue, simulation_depth=4, time_limit=100.0, rollouts=100000, exploration_weight=1.0):
        self.env = env
        self.known_queue = known_queue
        self.simulation_depth = simulation_depth
        self.time_limit = time_limit
        self.rollouts = rollouts
        self.exploration_weight = exploration_weight

    def best_action(self, board, tetromino):
        """
        Find the best move to make given current board state and tetromino.
        
        Args:
            board: Current board state
            tetromino: Current active tetromino
            
        Returns:
            int: Index of the best move to make
        """
        
        root = MCTSNode(
            self.env, board, tetromino, parent=None, move=None, queue_index=0, depth=0, known_queue=self.known_queue
        )
        
        rollouts = 0
        start_time = time.time()
        
        # Run MCTS iterations until the time limit or rollout limit is reached.
        while rollouts < self.rollouts and time.time() - start_time < self.time_limit:
            rollouts += 1
            node = self.select(root)
            simulation_result = self.rollout(node.board, node.tetromino, node.queue_index, node.depth)
            self.backpropagate(node, simulation_result)
        print(f"Rollouts: {rollouts} | Time: {time.time() - start_time:.2f}")
       
        # Choose the move at the root with the highest average reward.
        children_items = list(root.children.items())
        if children_items:
            moves, children = zip(*children_items)
            visits = np.array([child.visits for child in children])
            total_values = np.array([child.total_value for child in children])
            # Only compute averages where visits > 0. Otherwise assign -infinity.
            avg_values = np.where(visits > 0, total_values / visits, -np.inf)
            best_idx = np.argmax(avg_values)
            best_move = moves[best_idx]
            best_avg = avg_values[best_idx]
        else:
            best_move = None
            best_avg = float("-inf")
        
        # If no move was chosen, choose a random valid move.
        if best_move is None:
            valid_moves = valid_indices(Next(self.env).observation(self.env, board, tetromino))
            best_move = random.choice(valid_moves) if valid_moves else self.env.action_space.sample()
        
        print(f"MCTS best move value: {best_avg:.4f}")
        return best_move

    def select(self, node):
        # Traverse the tree until reaching a node that is not fully expanded or is terminal.
        while not self.is_terminal(node):
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = self.best_child(node)
        return node

    def best_child(self, node):
        """
        Select the most promising child node using the UCB formula.
        
        The UCB formula balances:
        - Exploitation: Nodes with high average value
        - Exploration: Nodes visited less frequently
        
        Args:
            node: Parent node
            
        Returns:
            MCTSNode: Most promising child node
        """
        
        moves = list(node.children.keys())
        visits = np.array([node.children[m].visits for m in moves])
        total_values = np.array([node.children[m].total_value for m in moves])
        
        # Compute exploitation and exploration terms 
        with np.errstate(divide='ignore', invalid='ignore'):
            exploitation = np.where(visits > 0, total_values / visits, 0)
            exploration = np.where(
                visits > 0,
                self.exploration_weight * np.sqrt(np.log(node.visits) / visits),
                float("inf")
            )
            
        scores = exploitation + exploration
        best_index = np.argmax(scores)
        return node.children[moves[best_index]]

    def rollout(self, board, tetromino, queue_index, depth):
        """
        Perform a random simulation from the given board state.
        
        Args:
            board: Current board state
            tetromino: Active tetromino
            queue_index: Position in the known tetromino queue
            depth: Current simulation depth
            
        Returns:
            float: Evaluated value of final board state
        """
        # If the simulation depth has been reached, evaluate the board using FeatureVector.
        if depth >= self.simulation_depth:
            return FeatureVector().value(board)
        nxt = Next(self.env)
        moves = nxt.observation(self.env, board, tetromino)
        valid_moves = valid_indices(moves)
        if not valid_moves:
            return float("-inf")
        move = random.choice(valid_moves)
        new_board = moves[move]
        # Determine the next tetromino for the rollout.
        if queue_index < len(self.known_queue):
            next_tet = self.env.env.unwrapped.TETROMINOES[self.known_queue[queue_index]]
            new_queue_index = queue_index + 1
        else:
            next_tet = random.choice(self.env.env.unwrapped.TETROMINOES)
            new_queue_index = queue_index
        return self.rollout(new_board, next_tet, new_queue_index, depth + 1)

    def backpropagate(self, node, value):
        """
        Update node statistics up the tree based on simulation results.
        
        Args:
            node: Node where simulation was performed
            value: Value obtained from simulation
        """
        # Propagate the simulation result up to the root.
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def is_terminal(self, node):
        # A node is terminal if the simulation depth is reached or if no legal moves exist.
        return node.depth >= self.simulation_depth or len(valid_indices(node.all_moves)) == 0

def simulate_lookahead(env, board, tet, upcoming):
    """
    Recursively simulate placements for the current tetromino and all upcoming tetrominoes.
    
    Args:
        env: The Tetris environment.
        board: Current board state.
        tet: The tetromino to place.
        upcoming: List (queue) of upcoming tetromino indices.
        
    Returns:
        A numerical evaluation of the final board state after simulating all moves.
    """
    nxt = Next(env)
    moves = nxt.observation(env, board, tet)
    valid_moves = valid_indices(moves)
    
    if not valid_moves:
        return float("-inf")
    
    # Base case: if there are no more upcoming pieces, evaluate the board.
    if not upcoming:
        return FeatureVector().value(board)
    
    best_value = float("-inf")
    for move in valid_moves:
        new_board = moves[move]
        next_tet = env.env.unwrapped.TETROMINOES[upcoming[0]]
        value = simulate_lookahead(env, new_board, next_tet, upcoming[1:])
        best_value = max(best_value, value)
    return best_value

def best_action_multistep(env, board, queue, tet):
    """
    Choose the best action by performing a multistep lookahead using the provided queue.
    
    Args:
        env: The Tetris environment.
        board: Current board state.
        queue: A list of upcoming tetromino indices.
        tet: The active tetromino.
        
    Returns:
        The index corresponding to the best move among the valid placements.
    """
    nxt = Next(env)
    moves = nxt.observation(env, board, tet)
    valid_moves = valid_indices(moves)
    
    if not valid_moves:
        return env.action_space.sample()
    
    best_val = float("-inf")
    best_move = valid_moves[0]
    for move in valid_moves:
        new_board = moves[move]
        if queue:
            value = simulate_lookahead(env, new_board, env.env.unwrapped.TETROMINOES[queue[0]], queue[1:])
        else:
            value = FeatureVector().value(new_board)
        if value > best_val:
            best_val = value
            best_move = move
    print("Multistep lookahead value:", best_val)
    return best_move

if __name__ == "__main__":
    # Parse command-line argument for number of lookahead tetriminos.
    parser = argparse.ArgumentParser(description="Tetris Multistep Lookahead")
    parser.add_argument(
        "--num_look_ahead",
        type=int,
        default=2,
        help="Number of tetriminos to look ahead. Must be between 1 and 4 (default: 1)"
    )
    parser.add_argument(
        "--mcts_rollouts", 
        type=int,
        help="MCTS rollouts per action"
    )
    parser.add_argument(
        "--mcts_time_limit", 
        type=float, 
        help="Time limit for MCTS in seconds"
    )
    parser.add_argument(
        "--mcts_exploration_weight", 
        type=float, 
        default=1.0,
        help="Exploration weight for UCB (default: 1.0)"
    )
    parser.add_argument(
        "--beam_width", 
        type=int, 
        default=10,
        help="Beam width (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="lookahead",
        choices=["mcts", "beam", "lookahead"],
        help="Strategy to use: mcts, beam, lookahead"
    )
    
    args = parser.parse_args()
    if args.strategy == "mcts":
        if args.mcts_rollouts is None and args.mcts_time_limit is None:
            print("Must give argument for either mcts_rollouts or mcts_time_limit")
            sys.exit(1)
    elif args.strategy == "lookahead":
        if args.num_look_ahead < 1 or args.num_look_ahead > 4:
            print("Error: num_look_ahead must be between 1 and 4")
            sys.exit(1)
    else:
        if args.beam_width < 1 or args.beam_width > 40:
            print("Error: beam_width must be between 1 and 40")
            sys.exit(1)
    
    print("Initializing Tetris Environment")
    # Initialize the environment and wrap it for grouped actions
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    env = GroupedActionsObservations(
        env,
        observation_wrappers=None,
        terminate_on_illegal_action=True
    )
    env.reset()
    observation = env.reset()[0]
    total_reward = 0
    terminated = False
    fet = FeatureVector()
    print("Active Tetromino:", env.env.unwrapped.active_tetromino)
    iteration = 0
    
    start_time = time.time()

    while not terminated:
        # print("Iteration:", iteration)
        # Get the full queue (expected to have 5 upcoming tetromino indices)
        queue = env.env.unwrapped.queue.get_queue()
        # print("Current lookahead queue:", queue[:args.num_look_ahead])
        if args.strategy == "lookahead":
            action = best_action_multistep(
                env,
                env.env.unwrapped.board,
                queue[:args.num_look_ahead],
                env.env.unwrapped.active_tetromino
            )
        elif args.strategy == "mcts":
            if args.mcts_time_limit is None:
                args.mcts_time_limit = 100.0
            if args.mcts_rollouts is None:
                args.mcts_rollouts = 100000
            mcts = MCTS(
                env, 
                known_queue=queue, 
                simulation_depth=args.num_look_ahead,
                time_limit=args.mcts_time_limit,
                rollouts=args.mcts_rollouts,
                exploration_weight=args.mcts_exploration_weight
            )
            
            action = mcts.best_action(
                env.env.unwrapped.board,
                env.env.unwrapped.active_tetromino
            )
        elif args.strategy == "beam":
            beam_search = BeamSearch(
                env,
                beam_width=args.beam_width,
                lookahead_steps=args.num_look_ahead
            )
        
            action = beam_search.best_action(
                env.env.unwrapped.board,
                env.env.unwrapped.active_tetromino,
                queue[:args.num_look_ahead],
            )
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        iteration += 1
        if iteration % 100 == 0:
            cur_time = time.time()
            hours = (cur_time - start_time) // 3600
            minutes = ((cur_time - start_time) % 3600) // 60
            print(f"Time: {hours:.0f}h {minutes:.0f}m")
            print(f"Total Reward so far: {total_reward}")
            print(f"Iteration: {iteration}")
        observation = next_observation

    print("Final Total Reward:", total_reward)
    
    # Optionally, display one of the final projected placements
    try:
        nxt = Next(env)
        final_moves = nxt.observation(env, env.env.unwrapped.board, env.env.unwrapped.active_tetromino)
        if len(final_moves) > 4:
            img = final_moves[4]
            plt.imshow(img)
            plt.axis("off")
            plt.show(block=False)
    except Exception as e:
        print(f"Warning: Could not display final board state: {e}")
