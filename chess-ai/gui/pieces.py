"""
Piece rendering for Chess AI GUI.

Supports:
- PNG images (preferred)
- Unicode character fallback
"""

import os
import chess
import pygame
from typing import Dict, Optional, Tuple


# Unicode chess pieces
UNICODE_PIECES = {
    (chess.KING, chess.WHITE): '\u2654',    # ♔
    (chess.QUEEN, chess.WHITE): '\u2655',   # ♕
    (chess.ROOK, chess.WHITE): '\u2656',    # ♖
    (chess.BISHOP, chess.WHITE): '\u2657',  # ♗
    (chess.KNIGHT, chess.WHITE): '\u2658',  # ♘
    (chess.PAWN, chess.WHITE): '\u2659',    # ♙
    (chess.KING, chess.BLACK): '\u265A',    # ♚
    (chess.QUEEN, chess.BLACK): '\u265B',   # ♛
    (chess.ROOK, chess.BLACK): '\u265C',    # ♜
    (chess.BISHOP, chess.BLACK): '\u265D',  # ♝
    (chess.KNIGHT, chess.BLACK): '\u265E',  # ♞
    (chess.PAWN, chess.BLACK): '\u265F',    # ♟
}

# File naming convention
PIECE_FILENAMES = {
    (chess.KING, chess.WHITE): 'w_king.png',
    (chess.QUEEN, chess.WHITE): 'w_queen.png',
    (chess.ROOK, chess.WHITE): 'w_rook.png',
    (chess.BISHOP, chess.WHITE): 'w_bishop.png',
    (chess.KNIGHT, chess.WHITE): 'w_knight.png',
    (chess.PAWN, chess.WHITE): 'w_pawn.png',
    (chess.KING, chess.BLACK): 'b_king.png',
    (chess.QUEEN, chess.BLACK): 'b_queen.png',
    (chess.ROOK, chess.BLACK): 'b_rook.png',
    (chess.BISHOP, chess.BLACK): 'b_bishop.png',
    (chess.KNIGHT, chess.BLACK): 'b_knight.png',
    (chess.PAWN, chess.BLACK): 'b_pawn.png',
}


class PieceRenderer:
    """
    Renders chess pieces using PNG images or Unicode fallback.

    Attributes:
        square_size: Size of each square in pixels
        assets_dir: Directory containing piece images
        use_images: Whether to use PNG images
    """

    def __init__(
        self,
        square_size: int = 60,
        assets_dir: str = 'assets/pieces',
    ):
        """
        Initialize piece renderer.

        Args:
            square_size: Size of each square in pixels
            assets_dir: Directory containing piece PNG files
        """
        self.square_size = square_size
        self.assets_dir = assets_dir
        self.images: Dict[Tuple[int, bool], pygame.Surface] = {}
        self.use_images = False
        self._font: Optional[pygame.font.Font] = None

        # Try to load images
        self._load_images()

    def _load_images(self):
        """Attempt to load piece images from assets directory."""
        if not os.path.isdir(self.assets_dir):
            print(f"Assets directory not found: {self.assets_dir}")
            print("Using Unicode fallback for pieces")
            return

        all_loaded = True
        for key, filename in PIECE_FILENAMES.items():
            path = os.path.join(self.assets_dir, filename)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    # Scale to square size
                    img = pygame.transform.smoothscale(
                        img, (self.square_size, self.square_size)
                    )
                    self.images[key] = img
                except pygame.error as e:
                    print(f"Failed to load {filename}: {e}")
                    all_loaded = False
            else:
                all_loaded = False

        self.use_images = all_loaded and len(self.images) == 12

        if self.use_images:
            print(f"Loaded 12 piece images from {self.assets_dir}")
        else:
            print(f"Some images missing, using Unicode fallback")
            self.images.clear()

    def resize(self, new_square_size: int):
        """
        Resize all piece images.

        Args:
            new_square_size: New square size in pixels
        """
        if new_square_size == self.square_size:
            return

        self.square_size = new_square_size

        if self.use_images:
            # Reload and rescale images
            self._load_images()

    def _get_font(self) -> pygame.font.Font:
        """Get or create font for Unicode rendering."""
        if self._font is None:
            # Try to use a font that supports chess symbols
            font_size = int(self.square_size * 0.85)
            try:
                # Try common fonts that support chess Unicode
                for font_name in ['Segoe UI Symbol', 'Arial Unicode MS', 'DejaVu Sans', None]:
                    try:
                        self._font = pygame.font.SysFont(font_name, font_size)
                        break
                    except:
                        continue
                if self._font is None:
                    self._font = pygame.font.Font(None, font_size)
            except:
                self._font = pygame.font.Font(None, font_size)
        return self._font

    def render_piece(
        self,
        piece: chess.Piece,
        surface: pygame.Surface,
        x: int,
        y: int,
    ):
        """
        Render a piece at the specified position.

        Args:
            piece: Chess piece to render
            surface: Pygame surface to draw on
            x: X coordinate (top-left)
            y: Y coordinate (top-left)
        """
        key = (piece.piece_type, piece.color)

        if self.use_images and key in self.images:
            # Draw PNG image
            surface.blit(self.images[key], (x, y))
        else:
            # Draw Unicode character
            self._render_unicode_piece(piece, surface, x, y)

    def _render_unicode_piece(
        self,
        piece: chess.Piece,
        surface: pygame.Surface,
        x: int,
        y: int,
    ):
        """Render piece using Unicode character."""
        key = (piece.piece_type, piece.color)
        char = UNICODE_PIECES.get(key, '?')

        font = self._get_font()

        # Render with outline for better visibility
        # First render shadow/outline
        outline_color = (0, 0, 0) if piece.color == chess.WHITE else (50, 50, 50)
        main_color = (255, 255, 255) if piece.color == chess.WHITE else (30, 30, 30)

        # Render main piece
        text_surface = font.render(char, True, main_color)

        # Center in square
        text_rect = text_surface.get_rect()
        text_rect.center = (x + self.square_size // 2, y + self.square_size // 2)

        # Draw outline by rendering slightly offset in multiple directions
        outline_surface = font.render(char, True, outline_color)
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            outline_rect = outline_surface.get_rect()
            outline_rect.center = (
                x + self.square_size // 2 + dx,
                y + self.square_size // 2 + dy,
            )
            surface.blit(outline_surface, outline_rect)

        # Draw main piece
        surface.blit(text_surface, text_rect)

    def get_piece_surface(self, piece: chess.Piece) -> pygame.Surface:
        """
        Get a surface containing just the piece.

        Useful for drag-and-drop.

        Args:
            piece: Chess piece

        Returns:
            Pygame surface with the piece
        """
        surface = pygame.Surface(
            (self.square_size, self.square_size),
            pygame.SRCALPHA,
        )
        self.render_piece(piece, surface, 0, 0)
        return surface


def create_piece_renderer(
    square_size: int = 60,
    assets_dir: str = 'assets/pieces',
) -> PieceRenderer:
    """
    Create a piece renderer.

    Args:
        square_size: Size of each square
        assets_dir: Directory for PNG assets

    Returns:
        Configured PieceRenderer instance
    """
    return PieceRenderer(square_size=square_size, assets_dir=assets_dir)


if __name__ == '__main__':
    # Test piece renderer
    pygame.init()

    # Create a test window
    screen = pygame.display.set_mode((480, 120))
    pygame.display.set_caption("Piece Renderer Test")

    renderer = PieceRenderer(square_size=60)

    # Draw all pieces
    pieces = [
        chess.Piece(chess.KING, chess.WHITE),
        chess.Piece(chess.QUEEN, chess.WHITE),
        chess.Piece(chess.ROOK, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.PAWN, chess.WHITE),
        chess.Piece(chess.KING, chess.BLACK),
        chess.Piece(chess.QUEEN, chess.BLACK),
        chess.Piece(chess.ROOK, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.PAWN, chess.BLACK),
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill((128, 128, 128))

        # Draw pieces in a row
        for i, piece in enumerate(pieces[:6]):
            x = i * 60
            pygame.draw.rect(screen, (240, 217, 181), (x, 0, 60, 60))
            renderer.render_piece(piece, screen, x, 0)

        for i, piece in enumerate(pieces[6:]):
            x = i * 60
            pygame.draw.rect(screen, (181, 136, 99), (x, 60, 60, 60))
            renderer.render_piece(piece, screen, x, 60)

        pygame.display.flip()

    pygame.quit()
    print("Piece renderer test completed!")
