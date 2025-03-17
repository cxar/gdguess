#!/usr/bin/env python3
"""
Visualization functions for interactive inference with the Grateful Dead show dating model.
"""

import numpy as np
import pygame


def initialize_pygame():
    """Initialize Pygame for visualization."""
    pygame.init()
    pygame.font.init()

    # Set up the window
    info = pygame.display.Info()
    width, height = min(1024, info.current_w), min(768, info.current_h)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Grateful Dead Show Dating")

    # Fonts
    title_font = pygame.font.SysFont("Arial", 28, bold=True)
    header_font = pygame.font.SysFont("Arial", 24, bold=True)
    text_font = pygame.font.SysFont("Arial", 20)

    return screen, width, height, title_font, header_font, text_font


def draw_results(
    screen, width, height, title_font, header_font, text_font, prediction, audio_buffer
):
    """Draw prediction results on the screen."""
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    BLUE = (50, 100, 200)
    GREEN = (50, 200, 100)

    # Background
    screen.fill(WHITE)

    # Title
    title = title_font.render("Grateful Dead Show Dating", True, BLACK)
    screen.blit(title, (width // 2 - title.get_width() // 2, 20))

    # Date prediction
    if prediction:
        date_text = header_font.render(
            f"Predicted Date: {prediction['predicted_date'].strftime('%B %d, %Y')}",
            True,
            BLUE,
        )
        screen.blit(date_text, (width // 2 - date_text.get_width() // 2, 70))

        era_text = header_font.render(f"Era: {prediction['era_name']}", True, BLUE)
        screen.blit(era_text, (width // 2 - era_text.get_width() // 2, 110))

        # Era probabilities
        prob_header = text_font.render("Era Probabilities:", True, BLACK)
        screen.blit(prob_header, (width // 4, 160))

        y_pos = 190
        for era, prob in prediction["era_probabilities"].items():
            era_prob_text = text_font.render(f"{era}: {prob}", True, BLACK)
            screen.blit(era_prob_text, (width // 4, y_pos))
            y_pos += 30

    # Draw audio waveform
    if audio_buffer is not None:
        # Downsample for display
        downsample_factor = 500
        downsampled = audio_buffer[::downsample_factor]

        # Scale for display
        if np.max(np.abs(downsampled)) > 0:
            downsampled = downsampled / np.max(np.abs(downsampled)) * (height // 4)

        # Draw waveform
        pygame.draw.line(screen, DARK_GRAY, (0, height - 100), (width, height - 100), 1)

        for i in range(1, len(downsampled)):
            x1 = (i - 1) * width / len(downsampled)
            y1 = height - 100 + downsampled[i - 1]
            x2 = i * width / len(downsampled)
            y2 = height - 100 + downsampled[i]
            pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), 2)

    # Instructions
    instructions = text_font.render("Press ESC to quit", True, BLACK)
    screen.blit(instructions, (width - instructions.get_width() - 20, height - 40))

    pygame.display.flip() 