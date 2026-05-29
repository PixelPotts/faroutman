#!/usr/bin/env python3
"""
Groovy Stick Figure Audio Visualizer
- Fat stick figure center screen
- Thick (~50px) spectral outlines spawn at the figure and expand outward
- When an outline leaves the screen it's replaced with a fresh one
- Driven by real-time FFT analysis
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


# ── Spectrum analysis ──────────────────────────────────────────────────────

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


def get_stickman_outline_points(cx, cy, scale=1.0, density=4):
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


# ── Color mapping ──────────────────────────────────────────────────────────

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


def spectrum_color(band_index, num_bands, intensity=1.0, time_phase=0.0):
    hue = (band_index / num_bands * 360 + time_phase) % 360
    v = min(1.0, 0.5 + intensity * 0.5)
    return hsv_to_rgb(hue, 1.0, v)


# ── Expand outline points outward from center ─────────────────────────────

def expand_points(points, cx, cy, factor, noise):
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
        v = noise[i % len(noise)]
        new_dist = dist * factor * v
        expanded.append((cx + nx_d * new_dist, cy + ny_d * new_dist))
    return expanded


def points_offscreen(points, W, H, margin=100):
    """Check if ALL points are outside the screen + margin."""
    for (x, y) in points:
        if -margin < x < W + margin and -margin < y < H + margin:
            return False
    return True


# ── Ring (expanding outline) ──────────────────────────────────────────────

class Ring:
    """A single expanding stickman outline that moves outward over time."""

    def __init__(self, band_idx, birth_time, speed, line_width, noise):
        self.band_idx = band_idx
        self.birth_time = birth_time
        self.speed = speed          # expansion speed (units/sec)
        self.line_width = line_width
        self.noise = noise          # per-point variance array
        self.expand = 1.0           # current expansion factor

    def update(self, dt, energy):
        # expand outward; energy boosts speed
        self.expand += self.speed * dt * (0.7 + energy * 1.5)

    def is_offscreen(self, base_points, cx, cy, W, H):
        if self.expand < 1.5:
            return False
        # sample a few points to check
        step = max(1, len(base_points) // 12)
        for i in range(0, len(base_points), step):
            px, py = base_points[i]
            dx, dy = px - cx, py - cy
            dist = math.hypot(dx, dy)
            nd = dist * self.expand
            nx = cx + (dx / max(dist, 1)) * nd
            ny = cy + (dy / max(dist, 1)) * nd
            if -200 < nx < W + 200 and -200 < ny < H + 200:
                return False
        return True


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
    base_points = get_stickman_outline_points(cx, cy, scale, density=4)
    stickman_parts = build_stickman_parts(cx, cy, scale)
    n_pts = len(base_points)

    # ── Ring pool ──
    MAX_RINGS = 20
    SPAWN_INTERVAL = 0.18  # seconds between new ring spawns
    AVG_LINE_WIDTH = 50

    def make_noise():
        return [random.uniform(0.88, 1.12) for _ in range(n_pts)]

    def spawn_ring(t, band_idx):
        lw = random.randint(35, 65)  # avg ~50
        speed = random.uniform(0.35, 0.6)
        return Ring(band_idx, t, speed, lw, make_noise())

    rings = []
    ring_counter = 0
    last_spawn = 0.0

    smooth_bands = np.zeros(NUM_BANDS)
    start_ticks = pygame.time.get_ticks()
    running = True
    prev_ticks = start_ticks

    while running:
        now_ticks = pygame.time.get_ticks()
        dt = (now_ticks - prev_ticks) / 1000.0
        dt = min(dt, 0.05)  # clamp
        prev_ticks = now_ticks

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

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
        t_phase = elapsed_s * 40  # color rotation speed

        # ── Spawn new rings ──
        # Spawn faster when energy is high
        spawn_rate = SPAWN_INTERVAL / (0.5 + energy * 1.5)
        if elapsed_s - last_spawn > spawn_rate and len(rings) < MAX_RINGS:
            band_idx = ring_counter % NUM_BANDS
            rings.append(spawn_ring(elapsed_s, band_idx))
            ring_counter += 1
            last_spawn = elapsed_s

        # ── Update rings ──
        for ring in rings:
            ring.update(dt, energy)

        # ── Cull offscreen rings ──
        rings = [r for r in rings if not r.is_offscreen(base_points, cx, cy, W, H)]

        # ── Draw ──
        bg_val = int(5 + energy * 20)
        screen.fill((bg_val, bg_val, int(bg_val * 1.3)))

        # Draw rings back to front (largest first)
        rings_sorted = sorted(rings, key=lambda r: -r.expand)

        for ring in rings_sorted:
            # Modulate noise over time for wobble
            t_off = elapsed_s * 1.8
            noise = [
                n * (1.0 + smooth_bands[(ring.band_idx + 2) % NUM_BANDS] * 0.25
                     * math.sin(i * 0.08 + t_off))
                for i, n in enumerate(ring.noise)
            ]

            expanded = expand_points(base_points, cx, cy, ring.expand, noise)

            # Fade as it gets further out
            age_factor = max(0.0, 1.0 - (ring.expand - 1.0) / 6.0)
            intensity = smooth_bands[ring.band_idx] * (0.3 + age_factor * 0.7)
            color = spectrum_color(ring.band_idx, NUM_BANDS, intensity, t_phase)

            # Scale line width: thicker when closer, maintain avg ~50
            lw = max(8, int(ring.line_width * max(0.4, age_factor)))

            if len(expanded) > 2:
                int_pts = [(int(x), int(y)) for x, y in expanded]
                # Glow layer
                glow_c = tuple(max(0, c // 4) for c in color)
                pygame.draw.polygon(screen, glow_c, int_pts, min(lw + 12, 80))
                # Main outline
                pygame.draw.polygon(screen, color, int_pts, lw)

        # ── Draw stickman on top ──
        # Filled dark silhouette
        if len(base_points) > 2:
            int_base = [(int(x), int(y)) for x, y in base_points]
            pygame.draw.polygon(screen, (10, 10, 15), int_base, 0)

        # Body parts with bright outline
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
