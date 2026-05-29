#!/usr/bin/env python3
"""
Groovy Stick Figure Audio Visualizer
- Fat stick figure center screen
- Expanding spectral outlines emanate from behind, colored by frequency band
- Driven by real-time audio spectrum analysis
"""

import pygame
import numpy as np
import soundfile as sf
import sys
import math
import random
from scipy.fft import rfft, rfftfreq

# ── Audio loading ──────────────────────────────────────────────────────────

def load_audio(path, target_sr=44100):
    """Load audio file, return mono float samples and sample rate."""
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    # simple resample if needed
    if sr != target_sr:
        ratio = target_sr / sr
        n_out = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n_out)
        data = np.interp(indices, np.arange(len(data)), data)
        sr = target_sr
    return data, sr


# ── Spectrum analysis ──────────────────────────────────────────────────────

NUM_BANDS = 8

def get_spectrum_bands(chunk, sr, num_bands=NUM_BANDS):
    """Return normalized energy in num_bands frequency bands."""
    if len(chunk) == 0:
        return np.zeros(num_bands)
    windowed = chunk * np.hanning(len(chunk))
    fft_mag = np.abs(rfft(windowed))
    freqs = rfftfreq(len(chunk), 1.0 / sr)
    # log-spaced band edges from 60Hz to 16kHz
    edges = np.logspace(np.log10(60), np.log10(16000), num_bands + 1)
    bands = np.zeros(num_bands)
    for i in range(num_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if mask.any():
            bands[i] = np.sqrt(np.mean(fft_mag[mask] ** 2))
    # normalize
    mx = bands.max()
    if mx > 0:
        bands = bands / mx
    return bands


# ── Stick figure geometry ──────────────────────────────────────────────────

def build_stickman_points(cx, cy, scale=1.0):
    """
    Return list of (points_list, line_width) tuples defining a fat stick man.
    Each element is a polyline or circle descriptor.
    All coords centered at (cx, cy).
    """
    s = scale
    parts = []

    # Head (circle approximation as polygon)
    head_r = 40 * s
    head_cy = cy - 160 * s
    head_pts = []
    for a in range(0, 361, 10):
        rad = math.radians(a)
        head_pts.append((cx + head_r * math.cos(rad), head_cy + head_r * math.sin(rad)))
    parts.append(('poly', head_pts, int(12 * s)))

    # Neck
    neck_top = cy - 120 * s
    neck_bot = cy - 100 * s
    parts.append(('line', [(cx, neck_top), (cx, neck_bot)], int(14 * s)))

    # Torso
    torso_top = cy - 100 * s
    torso_bot = cy + 40 * s
    parts.append(('line', [(cx, torso_top), (cx, torso_bot)], int(18 * s)))

    # Shoulders
    sh_y = cy - 80 * s
    sh_w = 70 * s
    parts.append(('line', [(cx - sh_w, sh_y), (cx + sh_w, sh_y)], int(14 * s)))

    # Left arm
    parts.append(('line', [
        (cx - sh_w, sh_y),
        (cx - sh_w - 20 * s, sh_y + 60 * s),
        (cx - sh_w - 10 * s, sh_y + 120 * s),
    ], int(12 * s)))

    # Right arm
    parts.append(('line', [
        (cx + sh_w, sh_y),
        (cx + sh_w + 20 * s, sh_y + 60 * s),
        (cx + sh_w + 10 * s, sh_y + 120 * s),
    ], int(12 * s)))

    # Hips
    hip_y = cy + 40 * s
    hip_w = 40 * s
    parts.append(('line', [(cx - hip_w, hip_y), (cx + hip_w, hip_y)], int(14 * s)))

    # Left leg
    parts.append(('line', [
        (cx - hip_w, hip_y),
        (cx - hip_w - 10 * s, hip_y + 80 * s),
        (cx - hip_w + 5 * s, hip_y + 160 * s),
    ], int(14 * s)))

    # Right leg
    parts.append(('line', [
        (cx + hip_w, hip_y),
        (cx + hip_w + 10 * s, hip_y + 80 * s),
        (cx + hip_w - 5 * s, hip_y + 160 * s),
    ], int(14 * s)))

    return parts


def get_stickman_outline_points(cx, cy, scale=1.0, density=3):
    """Get all outline points of the stickman as a single list for expansion."""
    parts = build_stickman_points(cx, cy, scale)
    all_points = []
    for part in parts:
        kind = part[0]
        pts = part[1]
        if kind == 'poly':
            all_points.extend(pts)
        elif kind == 'line':
            # interpolate along line segments
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                dist = math.hypot(x1 - x0, y1 - y0)
                n_steps = max(int(dist / density), 1)
                for t in range(n_steps):
                    frac = t / n_steps
                    all_points.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    return all_points


# ── Color mapping ──────────────────────────────────────────────────────────

def spectrum_color(band_index, num_bands, intensity=1.0, time_phase=0.0):
    """Map band index to a vibrant color with time-varying hue shift."""
    hue = (band_index / num_bands * 360 + time_phase) % 360
    # HSV to RGB with full saturation
    s, v = 1.0, min(1.0, 0.4 + intensity * 0.6)
    c = v * s
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = v - c
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ── Expand outline points outward from center ─────────────────────────────

def expand_points(points, cx, cy, factor, variance_arr=None):
    """Expand each point away from (cx, cy) by factor, with optional per-point variance."""
    expanded = []
    for i, (px, py) in enumerate(points):
        dx = px - cx
        dy = py - cy
        dist = math.hypot(dx, dy)
        if dist < 1:
            expanded.append((px, py))
            continue
        nx_d = dx / dist
        ny_d = dy / dist
        v = 1.0
        if variance_arr is not None:
            v = variance_arr[i % len(variance_arr)]
        new_dist = dist * factor * v
        expanded.append((cx + nx_d * new_dist, cy + ny_d * new_dist))
    return expanded


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    music_path = sys.argv[1] if len(sys.argv) > 1 else "sample_music.mp3"

    # Load audio
    print(f"Loading audio: {music_path}")
    samples, sr = load_audio(music_path)
    chunk_size = 2048
    total_chunks = len(samples) // chunk_size

    # Init pygame fullscreen
    pygame.init()
    pygame.mixer.init(frequency=sr, size=-16, channels=1, buffer=chunk_size)
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    pygame.display.set_caption("Stick Figure Visualizer")
    clock = pygame.time.Clock()

    # Play audio via pygame mixer
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    cx, cy = W // 2, H // 2 + 20
    scale = min(W, H) / 700.0
    base_points = get_stickman_outline_points(cx, cy, scale, density=4)
    stickman_parts = build_stickman_points(cx, cy, scale)

    # Number of expanding outline layers
    MAX_LAYERS = 14
    # Smoothed bands
    smooth_bands = np.zeros(NUM_BANDS)

    start_ticks = pygame.time.get_ticks()
    running = True

    # Pre-generate per-point noise offsets for organic feel
    noise_offsets = [random.uniform(0.85, 1.15) for _ in range(len(base_points))]

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Determine current audio position
        elapsed_ms = pygame.time.get_ticks() - start_ticks
        elapsed_s = elapsed_ms / 1000.0
        chunk_idx = int(elapsed_s * sr / chunk_size)

        if chunk_idx >= total_chunks:
            # Loop
            pygame.mixer.music.play()
            start_ticks = pygame.time.get_ticks()
            chunk_idx = 0

        # Get current audio chunk and analyze
        start_sample = chunk_idx * chunk_size
        chunk = samples[start_sample:start_sample + chunk_size]
        bands = get_spectrum_bands(chunk, sr)

        # Smooth
        smooth_bands = smooth_bands * 0.6 + bands * 0.4

        # Overall energy for bg pulse
        energy = np.mean(smooth_bands)

        # Time phase for color rotation
        t_phase = elapsed_s * 30  # degrees per second rotation

        # ── Draw ──
        # Dark background with subtle pulse
        bg_val = int(5 + energy * 25)
        screen.fill((bg_val, bg_val, int(bg_val * 1.2)))

        # Draw expanding outlines from back to front
        for layer in range(MAX_LAYERS, 0, -1):
            band_idx = layer % NUM_BANDS
            intensity = smooth_bands[band_idx]

            # Expansion factor: grows with layer and intensity
            base_expand = 1.0 + layer * 0.18
            expand = base_expand + intensity * layer * 0.12

            # Per-point variance modulated by a different band
            var_band = smooth_bands[(band_idx + 3) % NUM_BANDS]
            variance = [
                n * (1.0 + var_band * 0.3 * math.sin(i * 0.1 + elapsed_s * 2))
                for i, n in enumerate(noise_offsets)
            ]

            expanded = expand_points(base_points, cx, cy, expand, variance)

            # Line width from intensity
            lw = max(1, int(1 + intensity * 4))

            # Color from spectrum
            alpha_factor = max(0.15, 1.0 - layer / (MAX_LAYERS + 2))
            color = spectrum_color(band_idx, NUM_BANDS, intensity * alpha_factor, t_phase)

            # Draw as connected polygon outline
            if len(expanded) > 2:
                int_pts = [(int(x), int(y)) for x, y in expanded]
                # Draw glow (thicker, dimmer underneath)
                glow_color = tuple(max(0, c // 3) for c in color)
                pygame.draw.polygon(screen, glow_color, int_pts, max(1, lw + 2))
                pygame.draw.polygon(screen, color, int_pts, lw)

        # Draw the solid stickman on top (black fill with white outline)
        # First draw filled silhouette
        if len(base_points) > 2:
            int_base = [(int(x), int(y)) for x, y in base_points]
            pygame.draw.polygon(screen, (10, 10, 15), int_base, 0)  # filled

        # Draw stickman body parts with bright outlines
        body_color = (230, 230, 240)
        for part in stickman_parts:
            kind = part[0]
            pts = part[1]
            lw = part[2]
            int_pts = [(int(x), int(y)) for x, y in pts]
            if kind == 'poly':
                pygame.draw.polygon(screen, body_color, int_pts, max(2, lw))
                # fill dark
                pygame.draw.polygon(screen, (15, 15, 20), int_pts, 0)
                pygame.draw.polygon(screen, body_color, int_pts, max(2, lw))
            elif kind == 'line':
                pygame.draw.lines(screen, body_color, False, int_pts, max(2, lw))

        pygame.display.flip()
        clock.tick(60)

    pygame.mixer.music.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
