"""
Stats panel and controls for Chess AI GUI.

Features:
- Mode display (Spectator / Play / Training)
- Game statistics
- Control buttons
- Speed slider
- Stockfish review display
"""

import pygame
import sys
import os
from typing import Callable, Optional, Dict, Any
from enum import Enum, auto

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review import GameReview, MoveClassification


# Colors
PANEL_BG = (45, 45, 45)
PANEL_BORDER = (70, 70, 70)
TEXT_COLOR = (220, 220, 220)
TEXT_MUTED = (150, 150, 150)
BUTTON_BG = (60, 60, 60)
BUTTON_HOVER = (80, 80, 80)
BUTTON_ACTIVE = (100, 140, 100)
BUTTON_TEXT = (220, 220, 220)
SLIDER_BG = (60, 60, 60)
SLIDER_FILL = (100, 140, 100)
SLIDER_HANDLE = (180, 180, 180)

# Review colors
REVIEW_EXCELLENT = (100, 200, 100)  # Green
REVIEW_GOOD = (150, 200, 150)  # Light green
REVIEW_NEUTRAL = (180, 180, 180)  # Gray
REVIEW_INACCURACY = (220, 180, 100)  # Yellow
REVIEW_MISTAKE = (220, 140, 100)  # Orange
REVIEW_BLUNDER = (220, 80, 80)  # Red


class GameMode(Enum):
    """Game modes."""

    SPECTATOR = auto()
    PLAY = auto()
    TRAINING = auto()


class Button:
    """Simple button widget."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Optional[Callable] = None,
        toggle: bool = False,
    ):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.toggle = toggle
        self.active = False
        self.hovered = False
        self.font = pygame.font.Font(None, 24)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame event. Returns True if clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                if self.callback:
                    self.callback()
                return True
        return False

    def render(self, surface: pygame.Surface):
        """Render the button."""
        if self.active:
            color = BUTTON_ACTIVE
        elif self.hovered:
            color = BUTTON_HOVER
        else:
            color = BUTTON_BG

        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=4)

        text_surface = self.font.render(self.text, True, BUTTON_TEXT)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class Slider:
    """Simple slider widget."""

    def __init__(
        self,
        rect: pygame.Rect,
        min_val: float,
        max_val: float,
        initial: float,
        callback: Optional[Callable[[float], None]] = None,
    ):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.callback = callback
        self.dragging = False
        self.handle_radius = 8

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame event. Returns True if value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value(event.pos[0])
                return True
        return False

    def _update_value(self, x: int):
        """Update value based on x position."""
        rel_x = x - self.rect.x
        pct = max(0, min(1, rel_x / self.rect.width))
        self.value = self.min_val + pct * (self.max_val - self.min_val)
        if self.callback:
            self.callback(self.value)

    def render(self, surface: pygame.Surface):
        """Render the slider."""
        # Background
        pygame.draw.rect(surface, SLIDER_BG, self.rect, border_radius=4)

        # Fill
        pct = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(self.rect.width * pct)
        fill_rect = pygame.Rect(
            self.rect.x, self.rect.y, fill_width, self.rect.height
        )
        pygame.draw.rect(surface, SLIDER_FILL, fill_rect, border_radius=4)

        # Handle
        handle_x = self.rect.x + fill_width
        handle_y = self.rect.y + self.rect.height // 2
        pygame.draw.circle(
            surface, SLIDER_HANDLE, (handle_x, handle_y), self.handle_radius
        )


class StatsPanel:
    """
    Stats and controls panel.

    Displays game info and provides mode controls.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ):
        """
        Initialize stats panel.

        Args:
            x: X position
            y: Y position
            width: Panel width
            height: Panel height
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.width = width
        self.height = height

        # Fonts
        self.font_title = pygame.font.Font(None, 32)
        self.font_normal = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

        # State
        self.mode = GameMode.SPECTATOR
        self.turn_white = True
        self.is_paused = False
        self.player_color_white = True  # Player plays white

        # Stats
        self.session_games = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_games = 0

        # Speed (seconds per move)
        self.move_delay = 0.5

        # AI learning option
        self.ai_learns = True

        # Stockfish review
        self.review: Optional[GameReview] = None
        self.show_review = False
        self.stockfish_available = False

        # Callbacks
        self.on_new_game: Optional[Callable] = None
        self.on_mode_change: Optional[Callable[[GameMode], None]] = None
        self.on_color_change: Optional[Callable[[bool], None]] = None
        self.on_pause_toggle: Optional[Callable[[bool], None]] = None
        self.on_speed_change: Optional[Callable[[float], None]] = None

        # Create widgets
        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets."""
        self.buttons: Dict[str, Button] = {}
        self.sliders: Dict[str, Slider] = {}

        x = self.rect.x
        y = self.rect.y
        w = self.width
        pad = 10
        btn_h = 35

        # New game button
        self.buttons['new_game'] = Button(
            pygame.Rect(x + pad, y + 200, w - 2 * pad, btn_h),
            "New Game",
            callback=self._on_new_game,
        )

        # Mode buttons
        btn_w = (w - 3 * pad) // 2
        self.buttons['spectator'] = Button(
            pygame.Rect(x + pad, y + 250, btn_w, btn_h),
            "Spectator",
            callback=lambda: self._set_mode(GameMode.SPECTATOR),
            toggle=True,
        )
        self.buttons['spectator'].active = True

        self.buttons['play'] = Button(
            pygame.Rect(x + pad + btn_w + pad, y + 250, btn_w, btn_h),
            "Play",
            callback=lambda: self._set_mode(GameMode.PLAY),
            toggle=True,
        )

        # Color selection (for Play mode)
        self.buttons['white'] = Button(
            pygame.Rect(x + pad, y + 300, btn_w, btn_h),
            "White",
            callback=lambda: self._set_color(True),
            toggle=True,
        )
        self.buttons['white'].active = True

        self.buttons['black'] = Button(
            pygame.Rect(x + pad + btn_w + pad, y + 300, btn_w, btn_h),
            "Black",
            callback=lambda: self._set_color(False),
            toggle=True,
        )

        # Pause button
        self.buttons['pause'] = Button(
            pygame.Rect(x + pad, y + 350, w - 2 * pad, btn_h),
            "Pause",
            callback=self._toggle_pause,
            toggle=True,
        )

        # Speed slider
        self.sliders['speed'] = Slider(
            pygame.Rect(x + pad, y + 420, w - 2 * pad, 20),
            min_val=0.1,
            max_val=2.0,
            initial=0.5,
            callback=self._on_speed_change,
        )

        # AI learns checkbox (represented as button)
        self.buttons['ai_learns'] = Button(
            pygame.Rect(x + pad, y + 470, w - 2 * pad, btn_h),
            "AI Learns: ON",
            callback=self._toggle_ai_learns,
            toggle=True,
        )
        self.buttons['ai_learns'].active = True

    def _on_new_game(self):
        """Handle new game button."""
        if self.on_new_game:
            self.on_new_game()

    def _set_mode(self, mode: GameMode):
        """Set game mode."""
        self.mode = mode

        # Update button states
        self.buttons['spectator'].active = mode == GameMode.SPECTATOR
        self.buttons['play'].active = mode == GameMode.PLAY

        if self.on_mode_change:
            self.on_mode_change(mode)

    def _set_color(self, white: bool):
        """Set player color."""
        self.player_color_white = white
        self.buttons['white'].active = white
        self.buttons['black'].active = not white

        if self.on_color_change:
            self.on_color_change(white)

    def _toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.buttons['pause'].text = "Resume" if self.is_paused else "Pause"

        if self.on_pause_toggle:
            self.on_pause_toggle(self.is_paused)

    def _on_speed_change(self, value: float):
        """Handle speed change."""
        self.move_delay = value
        if self.on_speed_change:
            self.on_speed_change(value)

    def _toggle_ai_learns(self):
        """Toggle AI learning."""
        self.ai_learns = not self.ai_learns
        self.buttons['ai_learns'].text = f"AI Learns: {'ON' if self.ai_learns else 'OFF'}"

    def update_stats(
        self,
        session_games: int = None,
        white_wins: int = None,
        black_wins: int = None,
        draws: int = None,
        total_games: int = None,
    ):
        """Update displayed statistics."""
        if session_games is not None:
            self.session_games = session_games
        if white_wins is not None:
            self.white_wins = white_wins
        if black_wins is not None:
            self.black_wins = black_wins
        if draws is not None:
            self.draws = draws
        if total_games is not None:
            self.total_games = total_games

    def set_turn(self, white_turn: bool):
        """Set current turn indicator."""
        self.turn_white = white_turn

    def set_review(self, review: Optional[GameReview]):
        """Set the game review to display."""
        self.review = review
        self.show_review = review is not None

    def clear_review(self):
        """Clear the current review."""
        self.review = None
        self.show_review = False

    def set_stockfish_available(self, available: bool):
        """Set whether Stockfish is available."""
        self.stockfish_available = available

    def handle_event(self, event: pygame.event.Event):
        """Handle pygame event."""
        for button in self.buttons.values():
            button.handle_event(event)
        for slider in self.sliders.values():
            slider.handle_event(event)

    def render(self, surface: pygame.Surface):
        """Render the panel."""
        x = self.rect.x
        y = self.rect.y
        w = self.width
        pad = 10

        # Background
        pygame.draw.rect(surface, PANEL_BG, self.rect)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=2)

        # Title
        title = self.font_title.render("Chess AI", True, TEXT_COLOR)
        surface.blit(title, (x + pad, y + pad))

        # Mode
        mode_text = f"Mode: {self.mode.name.title()}"
        mode_surface = self.font_normal.render(mode_text, True, TEXT_COLOR)
        surface.blit(mode_surface, (x + pad, y + 50))

        # Turn indicator
        turn_text = f"Turn: {'White' if self.turn_white else 'Black'}"
        turn_surface = self.font_normal.render(turn_text, True, TEXT_COLOR)
        surface.blit(turn_surface, (x + pad, y + 75))

        # Session stats
        stats_y = y + 110
        session_text = f"Session: {self.session_games} games"
        session_surface = self.font_normal.render(session_text, True, TEXT_COLOR)
        surface.blit(session_surface, (x + pad, stats_y))

        # Win/loss/draw
        score_text = f"W: {self.white_wins}  B: {self.black_wins}  D: {self.draws}"
        score_surface = self.font_normal.render(score_text, True, TEXT_COLOR)
        surface.blit(score_surface, (x + pad, stats_y + 25))

        # Total games
        total_text = f"Total games: {self.total_games}"
        total_surface = self.font_small.render(total_text, True, TEXT_MUTED)
        surface.blit(total_surface, (x + pad, stats_y + 50))

        # Render buttons
        for button in self.buttons.values():
            button.render(surface)

        # Speed label
        speed_label = f"Speed: {self.move_delay:.1f}s/move"
        speed_surface = self.font_small.render(speed_label, True, TEXT_COLOR)
        surface.blit(speed_surface, (x + pad, y + 400))

        # Render sliders
        for slider in self.sliders.values():
            slider.render(surface)

        # Stockfish status
        sf_y = y + 500
        if self.stockfish_available:
            sf_text = "Stockfish: Active"
            sf_color = REVIEW_GOOD
        else:
            sf_text = "Stockfish: Not found"
            sf_color = TEXT_MUTED
        sf_surface = self.font_small.render(sf_text, True, sf_color)
        surface.blit(sf_surface, (x + pad, sf_y))

        # Render review if available
        if self.show_review and self.review:
            self._render_review(surface, x + pad, sf_y + 25, w - 2 * pad)

        # Keyboard hints
        hints_y = y + self.height - 80
        hints = [
            "Space: Pause/Resume",
            "N: New Game",
            "S: Save Model",
            "Q: Quit",
        ]
        for i, hint in enumerate(hints):
            hint_surface = self.font_small.render(hint, True, TEXT_MUTED)
            surface.blit(hint_surface, (x + pad, hints_y + i * 18))

    def _render_review(self, surface: pygame.Surface, x: int, y: int, width: int):
        """Render the Stockfish review summary."""
        if not self.review:
            return

        # Review title
        title = self.font_small.render("Review:", True, TEXT_COLOR)
        surface.blit(title, (x, y))
        y += 20

        # Stats with colors
        stats = [
            (f"Excellents: {self.review.excellent_count}", REVIEW_EXCELLENT),
            (f"Bons: {self.review.good_count}", REVIEW_GOOD),
            (f"Imprecisions: {self.review.inaccuracy_count}", REVIEW_INACCURACY),
            (f"Erreurs: {self.review.mistake_count}", REVIEW_MISTAKE),
            (f"Gaffes: {self.review.blunder_count}", REVIEW_BLUNDER),
        ]

        for text, color in stats:
            text_surface = self.font_small.render(text, True, color)
            surface.blit(text_surface, (x, y))
            y += 16

        # Worst move
        worst = self.review.get_worst_move()
        if worst and worst.delta < -100:
            y += 4
            worst_text = f"Pire: #{worst.move_number}"
            worst_surface = self.font_small.render(worst_text, True, REVIEW_BLUNDER)
            surface.blit(worst_surface, (x, y))


if __name__ == '__main__':
    # Test panel
    pygame.init()

    screen = pygame.display.set_mode((300, 600))
    pygame.display.set_caption("Stats Panel Test")

    panel = StatsPanel(10, 10, 280, 580)
    panel.update_stats(
        session_games=15,
        white_wins=7,
        black_wins=6,
        draws=2,
        total_games=150,
    )

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            panel.handle_event(event)

        screen.fill((30, 30, 30))
        panel.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Panel test completed!")
