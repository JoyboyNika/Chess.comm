"""
Chess board rendering for Chess AI GUI.

Features:
- 8×8 board with alternating colors
- Piece rendering
- Selected square highlighting
- Last move highlighting
- Legal move indicators
- Board flipping for black perspective
"""

import chess
import pygame
from typing import Optional, List, Tuple, Set

from .pieces import PieceRenderer


# Colors
LIGHT_SQUARE = (240, 217, 181)  # Cream
DARK_SQUARE = (181, 136, 99)   # Brown
SELECTED_COLOR = (186, 202, 68, 180)  # Yellow-green, semi-transparent
LAST_MOVE_COLOR = (205, 210, 106, 150)  # Light yellow, semi-transparent
LEGAL_MOVE_COLOR = (100, 100, 100, 128)  # Gray dot
CHECK_COLOR = (255, 0, 0, 100)  # Red tint for check


class BoardRenderer:
    """
    Renders a chess board with pieces and highlights.

    Attributes:
        board_size: Total size of the board in pixels
        square_size: Size of each square
        flipped: Whether board is flipped (black perspective)
    """

    def __init__(
        self,
        board_size: int = 480,
        assets_dir: str = 'assets/pieces',
    ):
        """
        Initialize board renderer.

        Args:
            board_size: Total board size in pixels
            assets_dir: Directory for piece assets
        """
        self.board_size = board_size
        self.square_size = board_size // 8
        self.flipped = False

        # Create piece renderer
        self.piece_renderer = PieceRenderer(
            square_size=self.square_size,
            assets_dir=assets_dir,
        )

        # Highlight state
        self.selected_square: Optional[int] = None
        self.legal_moves: Set[int] = set()
        self.last_move: Optional[chess.Move] = None

        # Create surfaces
        self._create_surfaces()

    def _create_surfaces(self):
        """Create reusable surfaces for board rendering."""
        # Board surface (static)
        self.board_surface = pygame.Surface(
            (self.board_size, self.board_size)
        )
        self._draw_empty_board()

        # Highlight surfaces
        self.highlight_surface = pygame.Surface(
            (self.square_size, self.square_size),
            pygame.SRCALPHA,
        )

        # Legal move dot
        self.legal_move_dot = pygame.Surface(
            (self.square_size, self.square_size),
            pygame.SRCALPHA,
        )
        dot_radius = self.square_size // 6
        pygame.draw.circle(
            self.legal_move_dot,
            LEGAL_MOVE_COLOR,
            (self.square_size // 2, self.square_size // 2),
            dot_radius,
        )

        # Legal move capture ring (for squares with enemy pieces)
        self.legal_capture_ring = pygame.Surface(
            (self.square_size, self.square_size),
            pygame.SRCALPHA,
        )
        ring_radius = self.square_size // 2 - 4
        ring_width = 4
        pygame.draw.circle(
            self.legal_capture_ring,
            LEGAL_MOVE_COLOR,
            (self.square_size // 2, self.square_size // 2),
            ring_radius,
            ring_width,
        )

    def _draw_empty_board(self):
        """Draw the empty chess board pattern."""
        for rank in range(8):
            for file in range(8):
                x = file * self.square_size
                y = rank * self.square_size

                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE

                pygame.draw.rect(
                    self.board_surface,
                    color,
                    (x, y, self.square_size, self.square_size),
                )

    def set_flipped(self, flipped: bool):
        """Set board orientation."""
        self.flipped = flipped

    def square_to_coords(self, square: int) -> Tuple[int, int]:
        """
        Convert chess square to pixel coordinates.

        Args:
            square: Chess square (0-63)

        Returns:
            (x, y) pixel coordinates for top-left of square
        """
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.flipped:
            x = (7 - file) * self.square_size
            y = rank * self.square_size
        else:
            x = file * self.square_size
            y = (7 - rank) * self.square_size

        return x, y

    def coords_to_square(self, x: int, y: int) -> Optional[int]:
        """
        Convert pixel coordinates to chess square.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate

        Returns:
            Chess square (0-63) or None if outside board
        """
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return None

        file = x // self.square_size
        rank = y // self.square_size

        if self.flipped:
            file = 7 - file
            rank = rank
        else:
            rank = 7 - rank

        return chess.square(file, rank)

    def set_selected(self, square: Optional[int], legal_moves: List[chess.Move] = None):
        """
        Set the selected square and legal moves.

        Args:
            square: Selected square or None to clear
            legal_moves: List of legal moves from this square
        """
        self.selected_square = square

        if square is not None and legal_moves:
            self.legal_moves = {move.to_square for move in legal_moves}
        else:
            self.legal_moves = set()

    def set_last_move(self, move: Optional[chess.Move]):
        """Set the last move for highlighting."""
        self.last_move = move

    def clear_highlights(self):
        """Clear all highlights."""
        self.selected_square = None
        self.legal_moves = set()
        self.last_move = None

    def render(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        x_offset: int = 0,
        y_offset: int = 0,
    ):
        """
        Render the complete board with pieces and highlights.

        Args:
            surface: Pygame surface to draw on
            board: Chess board state
            x_offset: X offset for board position
            y_offset: Y offset for board position
        """
        # Draw base board
        surface.blit(self.board_surface, (x_offset, y_offset))

        # Draw highlights
        self._draw_highlights(surface, board, x_offset, y_offset)

        # Draw pieces
        self._draw_pieces(surface, board, x_offset, y_offset)

        # Draw legal move indicators
        self._draw_legal_moves(surface, board, x_offset, y_offset)

    def _draw_highlights(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        x_offset: int,
        y_offset: int,
    ):
        """Draw square highlights."""
        # Last move highlight
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.square_to_coords(sq)
                self.highlight_surface.fill(LAST_MOVE_COLOR)
                surface.blit(
                    self.highlight_surface,
                    (x + x_offset, y + y_offset),
                )

        # Selected square highlight
        if self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            self.highlight_surface.fill(SELECTED_COLOR)
            surface.blit(
                self.highlight_surface,
                (x + x_offset, y + y_offset),
            )

        # Check highlight
        if board.is_check():
            king_square = board.king(board.turn)
            if king_square is not None:
                x, y = self.square_to_coords(king_square)
                self.highlight_surface.fill(CHECK_COLOR)
                surface.blit(
                    self.highlight_surface,
                    (x + x_offset, y + y_offset),
                )

    def _draw_pieces(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        x_offset: int,
        y_offset: int,
    ):
        """Draw all pieces on the board."""
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                x, y = self.square_to_coords(square)
                self.piece_renderer.render_piece(
                    piece,
                    surface,
                    x + x_offset,
                    y + y_offset,
                )

    def _draw_legal_moves(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        x_offset: int,
        y_offset: int,
    ):
        """Draw legal move indicators."""
        for sq in self.legal_moves:
            x, y = self.square_to_coords(sq)

            # Use ring for captures, dot for empty squares
            if board.piece_at(sq):
                surface.blit(
                    self.legal_capture_ring,
                    (x + x_offset, y + y_offset),
                )
            else:
                surface.blit(
                    self.legal_move_dot,
                    (x + x_offset, y + y_offset),
                )

    def resize(self, new_size: int):
        """
        Resize the board.

        Args:
            new_size: New board size in pixels
        """
        self.board_size = new_size
        self.square_size = new_size // 8
        self.piece_renderer.resize(self.square_size)
        self._create_surfaces()


def create_board_renderer(
    board_size: int = 480,
    assets_dir: str = 'assets/pieces',
) -> BoardRenderer:
    """
    Create a board renderer.

    Args:
        board_size: Total board size in pixels
        assets_dir: Directory for piece assets

    Returns:
        Configured BoardRenderer instance
    """
    return BoardRenderer(board_size=board_size, assets_dir=assets_dir)


if __name__ == '__main__':
    # Test board renderer
    pygame.init()

    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Board Renderer Test")

    renderer = BoardRenderer(board_size=480)
    board = chess.Board()

    # Simulate some moves
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    renderer.set_last_move(board.peek())
    renderer.set_selected(chess.E5, list(board.legal_moves))

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                sq = renderer.coords_to_square(mx - 10, my - 10)
                if sq is not None:
                    piece = board.piece_at(sq)
                    if piece and piece.color == board.turn:
                        moves = [m for m in board.legal_moves if m.from_square == sq]
                        renderer.set_selected(sq, moves)
                    elif sq in renderer.legal_moves:
                        # Make move
                        for move in board.legal_moves:
                            if move.from_square == renderer.selected_square and move.to_square == sq:
                                board.push(move)
                                renderer.set_last_move(move)
                                renderer.set_selected(None)
                                break
                    else:
                        renderer.set_selected(None)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    renderer.set_flipped(not renderer.flipped)

        screen.fill((50, 50, 50))
        renderer.render(screen, board, 10, 10)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Board renderer test completed!")
