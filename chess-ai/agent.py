"""
Chess AI Agent with REINFORCE learning.

Handles:
- Move selection with illegal move masking
- REINFORCE policy gradient learning
- Episode memory for learning
"""

import chess
import torch
import torch.optim as optim
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from model import ChessNet, create_model
from utils import (
    encode_board,
    get_legal_move_mask,
    move_to_index,
    index_to_move,
    NUM_MOVES,
)


@dataclass
class EpisodeMemory:
    """Stores (state, action, log_prob) tuples for a single game."""

    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    colors: List[chess.Color] = field(default_factory=list)  # Which color made each move

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        color: chess.Color,
    ):
        """Add a transition to memory."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.colors.append(color)

    def clear(self):
        """Clear all memory."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.colors.clear()

    def __len__(self):
        return len(self.states)


class ChessAgent:
    """
    Chess AI agent using REINFORCE algorithm.

    Attributes:
        model: Neural network for move prediction
        optimizer: Adam optimizer
        device: Computation device
        memory: Episode memory for learning
        gamma: Discount factor for rewards
        learning_rate: Learning rate for optimizer
    """

    def __init__(
        self,
        model: Optional[ChessNet] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = None,
    ):
        """
        Initialize the chess agent.

        Args:
            model: Pre-trained model (optional)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Initialize or use provided model
        if model is None:
            self.model = create_model(self.device)
        else:
            self.model = model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Episode memory
        self.memory = EpisodeMemory()

        # Training stats
        self.total_games = 0
        self.total_updates = 0

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        store_for_learning: bool = True,
    ) -> chess.Move:
        """
        Select a move for the current position.

        Uses the policy network with illegal move masking.

        Args:
            board: Current board position
            temperature: Sampling temperature (higher = more random)
            store_for_learning: Whether to store this for later learning

        Returns:
            Selected chess move
        """
        self.model.eval()

        # Encode board state
        state = encode_board(board, self.device)
        state_batch = state.unsqueeze(0)

        # Get move probabilities
        with torch.no_grad():
            log_probs = self.model(state_batch)

        # Mask illegal moves
        legal_mask = get_legal_move_mask(board, self.device)

        # Apply mask (set illegal moves to very low probability)
        masked_log_probs = log_probs.clone()
        masked_log_probs[0, ~legal_mask] = float('-inf')

        # Apply temperature
        if temperature != 1.0:
            masked_log_probs = masked_log_probs / temperature

        # Convert to probabilities
        probs = torch.softmax(masked_log_probs, dim=-1)

        # Sample from distribution
        if store_for_learning:
            # Need gradients for learning
            self.model.train()
            log_probs_grad = self.model(state_batch)
            masked_log_probs_grad = log_probs_grad.clone()
            masked_log_probs_grad[0, ~legal_mask] = float('-inf')

            if temperature != 1.0:
                masked_log_probs_grad = masked_log_probs_grad / temperature

            probs_grad = torch.softmax(masked_log_probs_grad, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(probs_grad)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store in memory
            self.memory.add(
                state=state.detach(),
                action=action.item(),
                log_prob=log_prob,
                color=board.turn,
            )
        else:
            # Just sample without gradients
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        # Convert index to move
        move = index_to_move(action.item())

        return move

    def select_move_greedy(self, board: chess.Board) -> chess.Move:
        """
        Select the best move according to the policy (no sampling).

        Args:
            board: Current board position

        Returns:
            Best chess move
        """
        self.model.eval()

        state = encode_board(board, self.device)
        state_batch = state.unsqueeze(0)

        with torch.no_grad():
            log_probs = self.model(state_batch)

        legal_mask = get_legal_move_mask(board, self.device)
        masked_log_probs = log_probs.clone()
        masked_log_probs[0, ~legal_mask] = float('-inf')

        # Get best move
        best_action = masked_log_probs.argmax(dim=-1).item()
        return index_to_move(best_action)

    def learn(self, reward: float, color: Optional[chess.Color] = None) -> float:
        """
        Update the policy using REINFORCE algorithm.

        Should be called at the end of a game with the final reward.

        Args:
            reward: Final game reward (1.0 win, -1.0 loss, 0.0 draw)
            color: If specified, only update moves made by this color.
                   If None, update all moves (reward for white, -reward for black)

        Returns:
            Total loss value
        """
        if len(self.memory) == 0:
            return 0.0

        self.model.train()

        # Calculate returns for each move
        # For chess, we use the final reward, adjusted by color
        returns = []

        for i, move_color in enumerate(self.memory.colors):
            if color is not None and move_color != color:
                # Skip moves by the other color if we're only training one color
                returns.append(0.0)
            else:
                # Adjust reward based on which color made the move
                if move_color == chess.WHITE:
                    r = reward
                else:
                    r = -reward

                # Apply discount based on how many moves ago
                moves_from_end = len(self.memory) - i - 1
                discounted_r = r * (self.gamma ** moves_from_end)
                returns.append(discounted_r)

        returns = torch.tensor(returns, device=self.device)

        # Normalize returns (helps with training stability)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.memory.log_probs, returns):
            if R != 0:  # Skip if return is 0 (happens when filtering by color)
                policy_loss.append(-log_prob * R)

        if len(policy_loss) == 0:
            self.memory.clear()
            return 0.0

        # Backpropagation
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clear memory
        self.memory.clear()
        self.total_updates += 1

        return loss.item()

    def clear_memory(self):
        """Clear episode memory without learning."""
        self.memory.clear()

    def save(self, path: str):
        """Save model weights."""
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_games': self.total_games,
                'total_updates': self.total_updates,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_games = checkpoint.get('total_games', 0)
        self.total_updates = checkpoint.get('total_updates', 0)


if __name__ == '__main__':
    # Test the agent
    agent = ChessAgent()
    print(f"Agent created on device: {agent.device}")

    # Test move selection on initial position
    board = chess.Board()
    move = agent.select_move(board)
    print(f"Selected move on initial position: {move.uci()}")
    assert move in board.legal_moves, "Move should be legal!"

    # Test a few more moves
    board.push(move)
    move2 = agent.select_move(board)
    print(f"Black's response: {move2.uci()}")
    assert move2 in board.legal_moves, "Move should be legal!"

    # Test learning
    loss = agent.learn(1.0)  # Pretend white won
    print(f"Learning loss: {loss:.4f}")

    # Test greedy selection
    board = chess.Board()
    greedy_move = agent.select_move_greedy(board)
    print(f"Greedy move: {greedy_move.uci()}")

    print("\nAgent test passed!")
