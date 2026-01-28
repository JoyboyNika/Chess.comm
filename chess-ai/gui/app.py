"""
Main Chess AI application with Pygame GUI.

Features:
- Spectator mode: Watch AI play against itself
- Play mode: Play against the AI
- Training mode: Batch training with stats
- Real-time statistics
- Model persistence
"""

import sys
import os
import time
import chess
import pygame
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ChessAgent
from storage import Storage, get_storage
from training import play_game, TrainingStats
from utils import result_to_reward
from engine import StockfishEngine, is_stockfish_available, get_engine
from review import analyze_game, compute_rewards, GameReview

from .board import BoardRenderer
from .panels import StatsPanel, GameMode


# Window settings
WINDOW_WIDTH = 780
WINDOW_HEIGHT = 600
BOARD_SIZE = 520
PANEL_WIDTH = 240
BOARD_MARGIN = 20

# Colors
BG_COLOR = (35, 35, 35)

# Timing
AI_THINK_DELAY = 0.3  # Delay before AI moves in play mode


class ChessApp:
    """
    Main Chess AI application.

    Manages game state, rendering, and user interaction.
    """

    def __init__(
        self,
        agent: Optional[ChessAgent] = None,
        storage: Optional[Storage] = None,
        assets_dir: str = 'assets/pieces',
    ):
        """
        Initialize the application.

        Args:
            agent: Chess agent (created if not provided)
            storage: Storage handler (created if not provided)
            assets_dir: Directory for piece assets
        """
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Chess AI - REINFORCE Learning")

        # Create window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Initialize components
        self.agent = agent or ChessAgent()
        self.storage = storage or get_storage()

        # Try to load existing model
        if self.storage.model_exists():
            self.storage.load_model(self.agent)
            print(f"Loaded model (version {self.storage.model_version})")

        # Create renderers
        self.board_renderer = BoardRenderer(
            board_size=BOARD_SIZE,
            assets_dir=assets_dir,
        )
        self.panel = StatsPanel(
            x=BOARD_SIZE + BOARD_MARGIN * 2,
            y=BOARD_MARGIN,
            width=PANEL_WIDTH,
            height=WINDOW_HEIGHT - BOARD_MARGIN * 2,
        )

        # Game state
        self.board = chess.Board()
        self.mode = GameMode.SPECTATOR
        self.is_paused = False
        self.player_color = chess.WHITE
        self.game_over = False
        self.game_result: Optional[str] = None

        # Timing
        self.last_move_time = time.time()
        self.move_delay = 0.5
        self.ai_thinking = False
        self.ai_think_start = 0.0

        # Session stats
        self.session_stats = TrainingStats()

        # Stockfish engine for review
        self.engine: Optional[StockfishEngine] = None
        self.stockfish_available = is_stockfish_available()

        if self.stockfish_available:
            try:
                self.engine = get_engine()
                print("Stockfish engine initialized for post-game review")
            except Exception as e:
                print(f"Failed to initialize Stockfish: {e}")
                self.stockfish_available = False

        # Connect panel callbacks
        self._setup_callbacks()

        # Update panel with initial stats
        self._update_panel_stats()
        self.panel.set_stockfish_available(self.stockfish_available)

        # Running flag
        self.running = True

    def _setup_callbacks(self):
        """Set up panel callback functions."""
        self.panel.on_new_game = self._on_new_game
        self.panel.on_mode_change = self._on_mode_change
        self.panel.on_color_change = self._on_color_change
        self.panel.on_pause_toggle = self._on_pause_toggle
        self.panel.on_speed_change = self._on_speed_change

    def _on_new_game(self):
        """Start a new game."""
        # Learn from current game if it's over and AI learns is enabled
        if self.game_over and self.panel.ai_learns and len(self.agent.memory) > 0:
            reward = result_to_reward(self.game_result, chess.WHITE)
            self.agent.learn(reward)

        self.board = chess.Board()
        self.game_over = False
        self.game_result = None
        self.board_renderer.clear_highlights()
        self.last_move_time = time.time()
        self.ai_thinking = False
        self.agent.clear_memory()

        # Clear review from previous game
        self.panel.clear_review()

        # Flip board if playing as black
        if self.mode == GameMode.PLAY:
            self.board_renderer.set_flipped(not self.player_color)

    def _on_mode_change(self, mode: GameMode):
        """Handle mode change."""
        self.mode = mode
        self._on_new_game()

        if mode == GameMode.SPECTATOR:
            self.board_renderer.set_flipped(False)
        elif mode == GameMode.PLAY:
            self.board_renderer.set_flipped(not self.player_color)

    def _on_color_change(self, white: bool):
        """Handle player color change."""
        self.player_color = chess.WHITE if white else chess.BLACK
        self.board_renderer.set_flipped(not white)
        self._on_new_game()

    def _on_pause_toggle(self, paused: bool):
        """Handle pause toggle."""
        self.is_paused = paused

    def _on_speed_change(self, speed: float):
        """Handle speed change."""
        self.move_delay = speed

    def _update_panel_stats(self):
        """Update panel with current statistics."""
        db_stats = self.storage.get_game_stats()

        self.panel.update_stats(
            session_games=self.session_stats.games_played,
            white_wins=self.session_stats.white_wins,
            black_wins=self.session_stats.black_wins,
            draws=self.session_stats.draws,
            total_games=db_stats['total'],
        )
        self.panel.set_turn(self.board.turn == chess.WHITE)

    def _handle_game_over(self):
        """Handle end of game."""
        self.game_over = True
        self.game_result = self.board.result()

        # Update stats
        self.session_stats.update(self.game_result, len(self.board.move_stack))

        # Save game to database
        self.storage.save_game(
            result=self.game_result,
            moves=list(self.board.move_stack),
            board=self.board,
        )

        # Learn if enabled
        if self.panel.ai_learns:
            if self.stockfish_available and self.engine is not None:
                # Use Stockfish review for per-move rewards
                try:
                    moves = list(self.board.move_stack)
                    review = analyze_game(moves, self.engine)
                    rewards = compute_rewards(review.moves)
                    self.agent.learn_from_rewards(rewards)
                    self.panel.set_review(review)
                    print(f"Game reviewed: {review.blunder_count} blunders, "
                          f"{review.mistake_count} mistakes")
                except Exception as e:
                    print(f"Review failed: {e}, using simple reward")
                    reward = result_to_reward(self.game_result, chess.WHITE)
                    self.agent.learn(reward)
            else:
                # Use simple end-of-game reward
                reward = result_to_reward(self.game_result, chess.WHITE)
                self.agent.learn(reward)

            self.agent.total_games += 1

        self._update_panel_stats()

        # In spectator mode, auto-restart after delay
        if self.mode == GameMode.SPECTATOR:
            self.last_move_time = time.time()  # Will restart after delay

    def _make_ai_move(self):
        """Have the AI make a move."""
        move = self.agent.select_move(
            self.board,
            temperature=1.0,
            store_for_learning=self.panel.ai_learns,
        )
        self.board.push(move)
        self.board_renderer.set_last_move(move)
        self.board_renderer.set_selected(None)
        self.last_move_time = time.time()

        if self.board.is_game_over():
            self._handle_game_over()

    def _handle_player_click(self, pos: tuple):
        """Handle player click on board."""
        if self.game_over or self.ai_thinking:
            return

        # Only handle clicks in play mode when it's player's turn
        if self.mode != GameMode.PLAY:
            return
        if self.board.turn != self.player_color:
            return

        # Convert click to square
        board_x = pos[0] - BOARD_MARGIN
        board_y = pos[1] - BOARD_MARGIN
        square = self.board_renderer.coords_to_square(board_x, board_y)

        if square is None:
            return

        # If a piece is selected, try to move
        if self.board_renderer.selected_square is not None:
            # Check if this is a legal move
            from_sq = self.board_renderer.selected_square
            for move in self.board.legal_moves:
                if move.from_square == from_sq and move.to_square == square:
                    # Handle promotion
                    if self.board.piece_at(from_sq).piece_type == chess.PAWN:
                        if chess.square_rank(square) in [0, 7]:
                            move = chess.Move(from_sq, square, chess.QUEEN)

                    self.board.push(move)
                    self.board_renderer.set_last_move(move)
                    self.board_renderer.set_selected(None)

                    if self.board.is_game_over():
                        self._handle_game_over()
                    else:
                        # AI will respond
                        self.ai_thinking = True
                        self.ai_think_start = time.time()
                    return

            # Clicking elsewhere deselects
            self.board_renderer.set_selected(None)

        # Try to select a piece
        piece = self.board.piece_at(square)
        if piece and piece.color == self.board.turn:
            legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
            self.board_renderer.set_selected(square, legal_moves)
        else:
            self.board_renderer.set_selected(None)

    def _update(self):
        """Update game state."""
        current_time = time.time()

        # Update turn indicator
        self.panel.set_turn(self.board.turn == chess.WHITE)

        if self.game_over:
            # In spectator mode, auto-restart after delay
            if self.mode == GameMode.SPECTATOR and not self.is_paused:
                if current_time - self.last_move_time > 2.0:
                    self._on_new_game()
            return

        if self.is_paused:
            return

        # Handle AI moves
        if self.mode == GameMode.SPECTATOR:
            # AI plays both sides
            if current_time - self.last_move_time > self.move_delay:
                self._make_ai_move()

        elif self.mode == GameMode.PLAY:
            # AI responds when it's not player's turn
            if self.ai_thinking:
                if current_time - self.ai_think_start > AI_THINK_DELAY:
                    self._make_ai_move()
                    self.ai_thinking = False
            elif self.board.turn != self.player_color:
                self.ai_thinking = True
                self.ai_think_start = current_time

    def _render(self):
        """Render the application."""
        # Clear screen
        self.screen.fill(BG_COLOR)

        # Render board
        self.board_renderer.render(
            self.screen,
            self.board,
            BOARD_MARGIN,
            BOARD_MARGIN,
        )

        # Render game over overlay
        if self.game_over:
            self._render_game_over()

        # Render panel
        self.panel.render(self.screen)

        # Update display
        pygame.display.flip()

    def _render_game_over(self):
        """Render game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (BOARD_MARGIN, BOARD_MARGIN))

        # Result text
        font = pygame.font.Font(None, 48)

        if self.game_result == '1-0':
            text = "White Wins!"
        elif self.game_result == '0-1':
            text = "Black Wins!"
        else:
            text = "Draw!"

        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(
            center=(BOARD_MARGIN + BOARD_SIZE // 2, BOARD_MARGIN + BOARD_SIZE // 2)
        )
        self.screen.blit(text_surface, text_rect)

        # Sub-text
        if self.board.is_checkmate():
            sub_text = "Checkmate"
        elif self.board.is_stalemate():
            sub_text = "Stalemate"
        elif self.board.is_insufficient_material():
            sub_text = "Insufficient Material"
        elif self.board.is_fifty_moves():
            sub_text = "50-Move Rule"
        elif self.board.is_repetition():
            sub_text = "Repetition"
        else:
            sub_text = ""

        if sub_text:
            sub_font = pygame.font.Font(None, 32)
            sub_surface = sub_font.render(sub_text, True, (200, 200, 200))
            sub_rect = sub_surface.get_rect(
                center=(
                    BOARD_MARGIN + BOARD_SIZE // 2,
                    BOARD_MARGIN + BOARD_SIZE // 2 + 40,
                )
            )
            self.screen.blit(sub_surface, sub_rect)

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.panel._toggle_pause()
                elif event.key == pygame.K_n:
                    self._on_new_game()
                elif event.key == pygame.K_s:
                    self._save_model()
                elif event.key == pygame.K_f:
                    self.board_renderer.set_flipped(not self.board_renderer.flipped)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_player_click(event.pos)

            # Pass event to panel
            self.panel.handle_event(event)

    def _save_model(self):
        """Save the current model."""
        path = self.storage.save_model(self.agent)
        print(f"Model saved to {path}")

    def run(self):
        """Run the main application loop."""
        print("Chess AI started. Press Q to quit.")

        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(60)

        # Save model on exit
        self._save_model()
        pygame.quit()


def run_app(
    agent: Optional[ChessAgent] = None,
    storage: Optional[Storage] = None,
    assets_dir: str = 'assets/pieces',
):
    """
    Run the Chess AI application.

    Args:
        agent: Optional pre-configured agent
        storage: Optional storage handler
        assets_dir: Directory for piece assets
    """
    app = ChessApp(agent=agent, storage=storage, assets_dir=assets_dir)
    app.run()


if __name__ == '__main__':
    run_app()
