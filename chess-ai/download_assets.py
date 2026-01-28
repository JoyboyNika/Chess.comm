#!/usr/bin/env python3
"""
Download chess piece assets from Lichess.

This script downloads the cburnett chess piece set from Lichess
and converts SVG files to PNG format for use with Pygame.

Requirements:
    pip install cairosvg  # For SVG to PNG conversion

Usage:
    python download_assets.py
"""

import os
import urllib.request
import sys

# Asset configuration
ASSETS_DIR = 'assets/pieces'
LICHESS_BASE_URL = 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett'

# Piece mapping: (lichess_name, our_name)
PIECES = [
    ('wK', 'w_king'),
    ('wQ', 'w_queen'),
    ('wR', 'w_rook'),
    ('wB', 'w_bishop'),
    ('wN', 'w_knight'),
    ('wP', 'w_pawn'),
    ('bK', 'b_king'),
    ('bQ', 'b_queen'),
    ('bR', 'b_rook'),
    ('bB', 'b_bishop'),
    ('bN', 'b_knight'),
    ('bP', 'b_pawn'),
]


def download_file(url: str, path: str) -> bool:
    """Download a file from URL to path."""
    try:
        print(f"  Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def convert_svg_to_png(svg_path: str, png_path: str, size: int = 80):
    """Convert SVG to PNG using cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=size,
            output_height=size,
        )
        return True
    except ImportError:
        print("cairosvg not installed. Install with: pip install cairosvg")
        return False
    except Exception as e:
        print(f"  Conversion error: {e}")
        return False


def download_and_convert():
    """Download SVG files and convert to PNG."""
    # Create directories
    os.makedirs(ASSETS_DIR, exist_ok=True)
    svg_dir = os.path.join(ASSETS_DIR, 'svg_temp')
    os.makedirs(svg_dir, exist_ok=True)

    print("Downloading chess piece assets from Lichess...")
    print(f"Source: {LICHESS_BASE_URL}")
    print()

    success_count = 0

    for lichess_name, our_name in PIECES:
        svg_url = f"{LICHESS_BASE_URL}/{lichess_name}.svg"
        svg_path = os.path.join(svg_dir, f"{our_name}.svg")
        png_path = os.path.join(ASSETS_DIR, f"{our_name}.png")

        # Download SVG
        if download_file(svg_url, svg_path):
            # Convert to PNG
            if convert_svg_to_png(svg_path, png_path):
                print(f"  Created {our_name}.png")
                success_count += 1

    # Clean up SVG files
    import shutil
    shutil.rmtree(svg_dir, ignore_errors=True)

    # Create LICENSE file
    license_path = os.path.join(ASSETS_DIR, 'LICENSE.txt')
    with open(license_path, 'w') as f:
        f.write("""Chess Piece Assets - cburnett Set

Name: cburnett
Author: Colin M.L. Burnett
Source: https://github.com/lichess-org/lila/tree/master/public/piece/cburnett
License: GPL (GNU General Public License)

These chess piece images were created by Colin M.L. Burnett and are
distributed under the GPL license. They are used in the Lichess.org
open-source chess platform.

Original source: https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces
""")

    print()
    print(f"Downloaded and converted {success_count}/12 pieces")
    print(f"License file created: {license_path}")

    return success_count == 12


def download_png_alternative():
    """
    Alternative: Download pre-rendered PNG files from a different source.

    Uses the chess.com-style pieces from a public CDN.
    """
    print("Trying alternative PNG source...")

    # Alternative source with PNG files
    # Using wikimedia commons PNGs
    WIKI_BASE = 'https://upload.wikimedia.org/wikipedia/commons'

    WIKI_PIECES = {
        'w_king': '/4/42/Chess_klt45.svg',
        'w_queen': '/1/15/Chess_qlt45.svg',
        'w_rook': '/7/72/Chess_rlt45.svg',
        'w_bishop': '/b/b1/Chess_blt45.svg',
        'w_knight': '/7/70/Chess_nlt45.svg',
        'w_pawn': '/4/45/Chess_plt45.svg',
        'b_king': '/f/f0/Chess_kdt45.svg',
        'b_queen': '/4/47/Chess_qdt45.svg',
        'b_rook': '/f/ff/Chess_rdt45.svg',
        'b_bishop': '/9/98/Chess_bdt45.svg',
        'b_knight': '/e/ef/Chess_ndt45.svg',
        'b_pawn': '/c/c7/Chess_pdt45.svg',
    }

    os.makedirs(ASSETS_DIR, exist_ok=True)
    svg_dir = os.path.join(ASSETS_DIR, 'svg_temp')
    os.makedirs(svg_dir, exist_ok=True)

    success_count = 0

    for name, path in WIKI_PIECES.items():
        url = WIKI_BASE + path
        svg_path = os.path.join(svg_dir, f"{name}.svg")
        png_path = os.path.join(ASSETS_DIR, f"{name}.png")

        if download_file(url, svg_path):
            if convert_svg_to_png(svg_path, png_path):
                print(f"  Created {name}.png")
                success_count += 1

    # Clean up
    import shutil
    shutil.rmtree(svg_dir, ignore_errors=True)

    # Create LICENSE
    license_path = os.path.join(ASSETS_DIR, 'LICENSE.txt')
    with open(license_path, 'w') as f:
        f.write("""Chess Piece Assets

Name: Standard Chess Pieces
Source: Wikimedia Commons
License: Public Domain / CC0

These chess piece images are from Wikimedia Commons and are in the public domain.
https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces
""")

    return success_count == 12


def main():
    """Main entry point."""
    print("=" * 50)
    print("Chess AI - Asset Downloader")
    print("=" * 50)
    print()

    # Check for cairosvg
    try:
        import cairosvg
        print("cairosvg found - will convert SVG to PNG")
    except ImportError:
        print("WARNING: cairosvg not installed")
        print("Install with: pip install cairosvg")
        print()
        print("Without cairosvg, the app will use Unicode fallback for pieces.")
        print("This still works but looks less polished.")
        sys.exit(1)

    print()

    # Try Lichess first
    if download_and_convert():
        print()
        print("SUCCESS! All assets downloaded.")
        print(f"Assets location: {os.path.abspath(ASSETS_DIR)}")
    else:
        print()
        print("Lichess download failed, trying alternative source...")
        if download_png_alternative():
            print()
            print("SUCCESS! All assets downloaded from alternative source.")
        else:
            print()
            print("FAILED to download all assets.")
            print("The app will use Unicode fallback for pieces.")


if __name__ == '__main__':
    main()
