"""Intuition network - Neural network for move selection without tree search.

This network learns to evaluate positions and suggest moves based purely on
pattern recognition, mimicking human intuition rather than calculation.
"""

import torch
import torch.nn as nn


class IntuitionNetwork(nn.Module):
    """Neural network that develops chess intuition through pattern learning.

    Unlike traditional chess engines that use tree search (minimax, MCTS),
    this network learns to "feel" good moves through accumulated pattern
    exposure during post-hoc review sessions.

    Architecture (to be implemented):
    - Input: Board representation (piece positions, castling rights, etc.)
    - Hidden: Pattern recognition layers
    - Output: Move probabilities + position evaluation
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 4):
        """Initialize the intuition network.

        Args:
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Placeholder - actual architecture in next ticket
        self._initialized = False

    def forward(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            board_tensor: Encoded board position

        Returns:
            Tuple of (move_probabilities, position_evaluation)
        """
        raise NotImplementedError("Network architecture to be implemented")

    def select_move(self, board_tensor: torch.Tensor, temperature: float = 1.0) -> int:
        """Select a move based on network output.

        Args:
            board_tensor: Encoded board position
            temperature: Sampling temperature (higher = more random)

        Returns:
            Index of selected move
        """
        raise NotImplementedError("Move selection to be implemented")

    def save(self, path: str) -> None:
        """Save network weights to file."""
        raise NotImplementedError("Save to be implemented")

    def load(self, path: str) -> None:
        """Load network weights from file."""
        raise NotImplementedError("Load to be implemented")


def get_device() -> torch.device:
    """Get the best available device (MPS for M3, CUDA, or CPU).

    Returns:
        torch.device configured for optimal performance
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
