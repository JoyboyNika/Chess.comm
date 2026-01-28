"""
Neural network model for Chess AI.

Architecture:
- Input: 768 (12 pieces × 64 squares)
- Hidden: 512 → 256
- Output: 4672 (all possible UCI moves)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(nn.Module):
    """
    Policy network for chess move prediction.

    Uses a simple MLP architecture with two hidden layers.
    Output represents probability distribution over all possible UCI moves.
    """

    INPUT_SIZE = 768  # 12 piece types × 64 squares
    HIDDEN1_SIZE = 512
    HIDDEN2_SIZE = 256
    OUTPUT_SIZE = 4672  # All possible UCI moves

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(self.INPUT_SIZE, self.HIDDEN1_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN1_SIZE, self.HIDDEN2_SIZE)
        self.fc3 = nn.Linear(self.HIDDEN2_SIZE, self.OUTPUT_SIZE)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 768)

        Returns:
            Output tensor of shape (batch_size, 4672) with log probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

    def get_move_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get move probabilities (not log probabilities).

        Args:
            x: Input tensor of shape (batch_size, 768)

        Returns:
            Probability tensor of shape (batch_size, 4672)
        """
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.exp(log_probs)


def create_model(device: str = None) -> ChessNet:
    """
    Create a new ChessNet model.

    Args:
        device: Device to place model on ('cpu', 'cuda', 'mps', or None for auto)

    Returns:
        ChessNet model on the specified device
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model = ChessNet()
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test the model
    model = create_model()
    print(f"Model created on device: {next(model.parameters()).device}")
    print(f"Model architecture:\n{model}")

    # Test forward pass
    test_input = torch.randn(1, 768)
    if torch.backends.mps.is_available():
        test_input = test_input.to('mps')
    elif torch.cuda.is_available():
        test_input = test_input.to('cuda')

    output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 4672), f"Expected (1, 4672), got {output.shape}"
    print("\nModel test passed!")
