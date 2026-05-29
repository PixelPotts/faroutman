#!/usr/bin/env python3
"""
Groovy Stick Figure Audio Visualizer
- Fat stick figure center screen
- Contiguous ~50px concentric outlines, no gaps, uniform width
- New outlines born at the figure, push all others outward off screen
- Like rasterized stroke in Photoshop, continuously from inside out
"""

import pygame
import numpy as np
import soundfile as sf
import sys
import math
from scipy.fft import rfft, rfftfreq

# ── Audio ──────────────────────────────────────────────────────────────────

def load_audio(path, target_sr=44100):
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        n_out = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n_out)
        data = np.interp(indices, np.arange(len(data)), data)
        sr = target_sr
    return data, sr


NUM_BANDS = 8

def get_spectrum_bands(chunk, sr, num_bands=NUM_BANDS):
    if len(chunk) == 0:
        return np.zeros(num_bands)
    windowed = chunk * np.hanning(len(chunk))
    fft_mag = np.abs(rfft(windowed))
    freqs = rfftfreq(len(chunk), 1.0 / sr)
    edges = np.logspace(np.log10(60), np.log10(16000), num_bands + 1)
    bands = np.zeros(num_bands)
    for i in range(num_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if mask.any():
            bands[i] = np.sqrt(np.mean(fft_mag[mask] ** 2))
    mx = bands.max()
    if mx > 0:
        bands = bands / mx
    return bands


# ── Stick figure geometry ──────────────────────────────────────────────────

def build_stickman_parts(cx, cy, scale=1.0):
    s = scale
    parts = []
    # Head
    head_r = 40 * s
    head_cy = cy - 160 * s
    head_pts = []
    for a in range(0, 361, 10):
        rad = math.radians(a)
        head_pts.append((cx + head_r * math.cos(rad), head_cy + head_r * math.sin(rad)))
    parts.append(('poly', head_pts, int(12 * s)))
    # Neck
    parts.append(('line', [(cx, cy - 120 * s), (cx, cy - 100 * s)], int(14 * s)))
    # Torso
    parts.append(('line', [(cx, cy - 100 * s), (cx, cy + 40 * s)], int(18 * s)))
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
    parts = build_stickman_parts(cx, cy, scale)
    all_points = []
    for part in parts:
        kind, pts = part[0], part[1]
        if kind == 'poly':
            all_points.extend(pts)
        elif kind == 'line':
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                dist = math.hypot(x1 - x0, y1 - y0)
                n_steps = max(int(dist / density), 1)
                for t in range(n_steps):
                    frac = t / n_steps
                    all_points.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    return all_points


# ── Precompute unit directions from center for each outline point ─────────

def precompute_directions(base_points, cx, cy):
    """For each point, store (dx_unit, dy_unit, base_dist)."""
    dirs = []
    for (px, py) in base_points:
        dx = px - cx
        dy = py - cy
        dist = math.hypot(dx, dy)
        if dist < 1:
            dirs.append((0, 0, 0))
        else:
            dirs.append((dx / dist, dy / dist, dist))
    return dirs


def offset_ring(dirs, cx, cy, pixel_offset):
    """Push each point outward by pixel_offset pixels. Returns int tuples."""
    result = []
    for (ux, uy, base_dist) in dirs:
        d = base_dist + pixel_offset
        result.append((int(cx + ux * d), int(cy + uy * d)))
    return result


# ── Color ──────────────────────────────────────────────────────────────────

def hsv_to_rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    music_path = sys.argv[1] if len(sys.argv) > 1 else "sample_music.mp3"

    print(f"Loading audio: {music_path}")
    samples, sr = load_audio(music_path)
    chunk_size = 2048
    total_chunks = len(samples) // chunk_size

    pygame.init()
    pygame.mixer.init(frequency=sr, size=-16, channels=1, buffer=chunk_size)
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    pygame.display.set_caption("Stick Figure Visualizer")
    clock = pygame.time.Clock()

    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    cx, cy = W // 2, H // 2 + 20
    scale = min(W, H) / 700.0
    base_points = get_stickman_outline_points(cx, cy, scale, density=3)
    stickman_parts = build_stickman_parts(cx, cy, scale)
    dirs = precompute_directions(base_points, cx, cy)

    RING_WIDTH = 50
    # max distance from center to screen corner
    max_reach = math.hypot(W / 2, H / 2) + RING_WIDTH * 2

    smooth_bands = np.zeros(NUM_BANDS)
    phase = 0.0  # continuously growing pixel offset

    start_ticks = pygame.time.get_ticks()
    prev_ticks = start_ticks
    running = True

    while running:
        now_ticks = pygame.time.get_ticks()
        dt = min((now_ticks - prev_ticks) / 1000.0, 0.05)
        prev_ticks = now_ticks

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Audio analysis
        elapsed_ms = now_ticks - start_ticks
        elapsed_s = elapsed_ms / 1000.0
        chunk_idx = int(elapsed_s * sr / chunk_size)

        if chunk_idx >= total_chunks:
            pygame.mixer.music.play()
            start_ticks = now_ticks
            chunk_idx = 0
            elapsed_s = 0.0

        start_sample = chunk_idx * chunk_size
        chunk = samples[start_sample:start_sample + chunk_size]
        bands = get_spectrum_bands(chunk, sr)
        smooth_bands = smooth_bands * 0.55 + bands * 0.45
        energy = float(np.mean(smooth_bands))

        # Phase grows continuously — this is the "new outlines pushing from inside"
        # Speed driven by music energy
        base_speed = 80.0  # pixels per second base
        phase += base_speed * (0.4 + energy * 2.0) * dt

        # Color rotation over time
        t_phase = elapsed_s * 35

        # ── Draw ──
        bg_val = int(5 + energy * 18)
        screen.fill((bg_val, bg_val, int(bg_val * 1.2)))

        # The fractional offset of the innermost ring
        inner_offset = phase % RING_WIDTH
        # Which "generation" is the innermost ring
        wave_base = int(phase // RING_WIDTH)

        # Draw rings from outermost to innermost so inner paints over outer edges
        # First figure out how many rings we need
        n_rings = int((max_reach - inner_offset) / RING_WIDTH) + 2

        for i in range(n_rings - 1, -1, -1):
            # Pixel offset from the stickman outline for this ring's CENTER
            center_offset = inner_offset + i * RING_WIDTH

            if center_offset - RING_WIDTH / 2 > max_reach:
                continue

            # Which wave this ring belongs to (for color assignment)
            wave_num = wave_base + i
            band_idx = wave_num % NUM_BANDS
            intensity = smooth_bands[band_idx]

            # Hue from band, brightness from intensity
            hue = (band_idx / NUM_BANDS * 360 + t_phase) % 360
            val = min(1.0, 0.35 + intensity * 0.65)
            sat = min(1.0, 0.7 + intensity * 0.3)
            color = hsv_to_rgb(hue, sat, val)

            # Draw as filled region between inner and outer boundaries
            outer_offset = center_offset + RING_WIDTH / 2
            inner_off = max(0, center_offset - RING_WIDTH / 2)

            outer_pts = offset_ring(dirs, cx, cy, outer_offset)
            inner_pts = offset_ring(dirs, cx, cy, inner_off)

            # Build a closed band: outer forward + inner reversed
            band_poly = outer_pts + inner_pts[::-1]

            if len(band_poly) > 2:
                pygame.draw.polygon(screen, color, band_poly, 0)

        # ── Draw stickman on top ──
        # Filled dark silhouette
        if len(base_points) > 2:
            int_base = [(int(x), int(y)) for x, y in base_points]
            pygame.draw.polygon(screen, (10, 10, 15), int_base, 0)

        # Body parts
        body_color = (230, 230, 240)
        for part in stickman_parts:
            kind, pts, lw = part
            int_pts = [(int(x), int(y)) for x, y in pts]
            if kind == 'poly':
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
