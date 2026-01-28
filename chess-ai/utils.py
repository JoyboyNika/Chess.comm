"""
Utility functions for Chess AI.

Contains:
- Board encoding (position to tensor)
- Move encoding/decoding (UCI string to index and vice versa)
- Helper functions
"""

import chess
import torch
from typing import List, Dict, Tuple, Optional


# Piece to index mapping (0-11)
PIECE_TO_INDEX: Dict[chess.Piece, int] = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.KNIGHT, chess.WHITE): 1,
    chess.Piece(chess.BISHOP, chess.WHITE): 2,
    chess.Piece(chess.ROOK, chess.WHITE): 3,
    chess.Piece(chess.QUEEN, chess.WHITE): 4,
    chess.Piece(chess.KING, chess.WHITE): 5,
    chess.Piece(chess.PAWN, chess.BLACK): 6,
    chess.Piece(chess.KNIGHT, chess.BLACK): 7,
    chess.Piece(chess.BISHOP, chess.BLACK): 8,
    chess.Piece(chess.ROOK, chess.BLACK): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 10,
    chess.Piece(chess.KING, chess.BLACK): 11,
}


def _generate_all_moves() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Generate all possible UCI moves and create bidirectional mappings.

    Returns:
        Tuple of (move_to_index, index_to_move) dictionaries
    """
    moves = []

    # Regular moves: from any square to any square
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq != to_sq:
                from_name = chess.square_name(from_sq)
                to_name = chess.square_name(to_sq)
                moves.append(f"{from_name}{to_name}")

    # Promotion moves: pawns on 7th rank to 8th rank (white) or 2nd to 1st (black)
    promotion_pieces = ['q', 'r', 'b', 'n']

    # White promotions (rank 7 to 8)
    for from_file in range(8):
        from_sq = chess.square(from_file, 6)  # Rank 7
        for to_file in range(max(0, from_file - 1), min(8, from_file + 2)):
            to_sq = chess.square(to_file, 7)  # Rank 8
            from_name = chess.square_name(from_sq)
            to_name = chess.square_name(to_sq)
            for promo in promotion_pieces:
                moves.append(f"{from_name}{to_name}{promo}")

    # Black promotions (rank 2 to 1)
    for from_file in range(8):
        from_sq = chess.square(from_file, 1)  # Rank 2
        for to_file in range(max(0, from_file - 1), min(8, from_file + 2)):
            to_sq = chess.square(to_file, 0)  # Rank 1
            from_name = chess.square_name(from_sq)
            to_name = chess.square_name(to_sq)
            for promo in promotion_pieces:
                moves.append(f"{from_name}{to_name}{promo}")

    # Create mappings
    move_to_index = {move: idx for idx, move in enumerate(moves)}
    index_to_move = {idx: move for idx, move in enumerate(moves)}

    return move_to_index, index_to_move


# Global move mappings
MOVE_TO_INDEX, INDEX_TO_MOVE = _generate_all_moves()
NUM_MOVES = len(MOVE_TO_INDEX)


def encode_board(board: chess.Board, device: str = 'cpu') -> torch.Tensor:
    """
    Encode a chess board position as a tensor.

    Uses a 12×64 = 768 dimensional representation:
    - 12 piece types (6 white + 6 black)
    - 64 squares

    Args:
        board: Chess board position
        device: Device to place tensor on

    Returns:
        Tensor of shape (768,) with 1s where pieces are located
    """
    encoding = torch.zeros(12, 64, device=device)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = PIECE_TO_INDEX[piece]
            encoding[piece_idx, square] = 1.0

    return encoding.flatten()


def encode_board_batch(boards: List[chess.Board], device: str = 'cpu') -> torch.Tensor:
    """
    Encode multiple board positions as a batch tensor.

    Args:
        boards: List of chess board positions
        device: Device to place tensor on

    Returns:
        Tensor of shape (batch_size, 768)
    """
    batch = torch.stack([encode_board(b, device) for b in boards])
    return batch


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to its index in the output vector.

    Args:
        move: Chess move

    Returns:
        Index in the 4672-dimensional output vector
    """
    uci = move.uci()
    if uci in MOVE_TO_INDEX:
        return MOVE_TO_INDEX[uci]
    # Fallback for moves not in our mapping (shouldn't happen with legal moves)
    raise ValueError(f"Move {uci} not found in move mapping")


def index_to_move(index: int) -> chess.Move:
    """
    Convert an index to a chess move.

    Args:
        index: Index in the output vector

    Returns:
        Chess move
    """
    if index not in INDEX_TO_MOVE:
        raise ValueError(f"Index {index} not found in move mapping")
    uci = INDEX_TO_MOVE[index]
    return chess.Move.from_uci(uci)


def get_legal_move_mask(board: chess.Board, device: str = 'cpu') -> torch.Tensor:
    """
    Create a mask for legal moves.

    Args:
        board: Chess board position
        device: Device to place tensor on

    Returns:
        Boolean tensor of shape (4672,) with True for legal moves
    """
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool, device=device)

    for move in board.legal_moves:
        try:
            idx = move_to_index(move)
            mask[idx] = True
        except ValueError:
            # Move not in our mapping, skip it
            pass

    return mask


def get_legal_move_indices(board: chess.Board) -> List[int]:
    """
    Get indices of all legal moves.

    Args:
        board: Chess board position

    Returns:
        List of indices for legal moves
    """
    indices = []
    for move in board.legal_moves:
        try:
            idx = move_to_index(move)
            indices.append(idx)
        except ValueError:
            pass
    return indices


def get_game_result(board: chess.Board) -> Optional[str]:
    """
    Get the result of a finished game.

    Args:
        board: Chess board position

    Returns:
        '1-0' for white win, '0-1' for black win, '1/2-1/2' for draw, None if game ongoing
    """
    if not board.is_game_over():
        return None

    result = board.result()
    return result


def result_to_reward(result: str, color: chess.Color) -> float:
    """
    Convert game result to reward for a given color.

    Args:
        result: Game result string ('1-0', '0-1', '1/2-1/2')
        color: Color to get reward for

    Returns:
        1.0 for win, -1.0 for loss, 0.0 for draw
    """
    if result == '1-0':
        return 1.0 if color == chess.WHITE else -1.0
    elif result == '0-1':
        return -1.0 if color == chess.WHITE else 1.0
    else:  # Draw
        return 0.0


def board_to_fen(board: chess.Board) -> str:
    """Get FEN string for a board position."""
    return board.fen()


def fen_to_board(fen: str) -> chess.Board:
    """Create a board from FEN string."""
    return chess.Board(fen)


if __name__ == '__main__':
    # Test utilities
    print(f"Number of possible moves in mapping: {NUM_MOVES}")

    # Test board encoding
    board = chess.Board()
    encoding = encode_board(board)
    print(f"Board encoding shape: {encoding.shape}")
    print(f"Non-zero elements: {(encoding != 0).sum().item()}")  # Should be 32 (all pieces)

    # Test legal move mask
    mask = get_legal_move_mask(board)
    print(f"Legal moves at start: {mask.sum().item()}")  # Should be 20

    # Test move encoding/decoding
    move = chess.Move.from_uci('e2e4')
    idx = move_to_index(move)
    decoded = index_to_move(idx)
    print(f"Move e2e4 -> index {idx} -> {decoded.uci()}")

    print("\nUtils test passed!")
