"""
Benchmark: MultiLine vs LineStack
==================================

Side-by-side comparison of real-time update performance between MultiLine
(single-buffer) and LineStack (per-line graphics).  Both subplots render
the same streaming sine-wave data so visual output is identical — only the
update cost differs.

The title of each subplot shows a rolling average of the time spent
inside its animation callback (the *update* cost, not the render cost).
"""

import os
import time
from collections import deque

import numpy as np
import fastplotlib as fpl


# ── tunables ────────────────────────────────────────────────────────
n_lines = 64
n_points = 2000
batch_size = 8  # new samples appended per frame

# ── shared state ────────────────────────────────────────────────────
phase = np.linspace(0, np.pi, n_lines, dtype=np.float32)[:, None]
offsets = np.linspace(0.0, 18.0, n_lines, dtype=np.float32)[:, None]
dt = 0.06
state = {"frame": 0}

# rolling window for timing (last 120 frames ≈ 2 s at 60 fps)
ml_times: deque[float] = deque(maxlen=120)
ls_times: deque[float] = deque(maxlen=120)


# ── figure with two subplots ───────────────────────────────────────
figure = fpl.Figure(
    shape=(1, 2),
    size=(1400, 600),
    names=[["MultiLine (single buffer)", "LineStack (per-line graphics)"]],
)


# ── left: MultiLine with ring-buffer scroll ────────────────────────
ml_data = np.zeros((n_lines, n_points), dtype=np.float32)
figure[0, 0].add_multi_line(
    ml_data,
    thickness=1.2,
    colors="w",
    name="signals",
    scroll=True,
    scroll_n_valid=0,
)


def _make_batch() -> np.ndarray:
    """Generate the next batch of y-values from the shared frame counter."""
    base_t = state["frame"] * batch_size * dt
    t_vals = base_t + dt * np.arange(1, batch_size + 1, dtype=np.float32)
    return np.sin(t_vals[None, :] + phase) + offsets  # (n_lines, batch_size)


def update_multiline(subplot):
    t0 = time.perf_counter()

    subplot["signals"].append_y(_make_batch())

    ml_times.append(time.perf_counter() - t0)
    if len(ml_times) >= 2:
        avg_us = np.mean(ml_times) * 1e6
        subplot.title = f"MultiLine — {avg_us:.0f} \u00b5s/frame"


# ── right: LineStack with per-line data updates ────────────────────
#
# LineStack doesn't have append_y / ring-buffer support, so we
# simulate the same streaming effect: maintain a ring buffer in
# Python, and overwrite all y-values every frame.
xs_ls = np.arange(n_points, dtype=np.float32)
ls_init = np.zeros((n_lines, n_points, 2), dtype=np.float32)
ls_init[:, :, 0] = xs_ls

figure[0, 1].add_line_stack(
    ls_init,
    thickness=1.2,
    colors="w",
    name="signals",
    separation=0.28,
)

# ring buffer state for LineStack side
_ls_buf = np.zeros((n_lines, n_points), dtype=np.float32)
_ls_head = 0
_ls_n_valid = 0


def update_linestack(subplot):
    global _ls_head, _ls_n_valid
    t0 = time.perf_counter()

    y_new = _make_batch()

    # write into ring buffer using vectorised indexing
    n_new = y_new.shape[1]
    write_idx = (_ls_head + np.arange(n_new)) % n_points
    _ls_buf[:, write_idx] = y_new
    _ls_head = (_ls_head + n_new) % n_points
    _ls_n_valid = min(_ls_n_valid + n_new, n_points)

    # reconstruct the logical (unwrapped) view — single fancy-index
    if _ls_n_valid < n_points:
        logical = np.zeros((n_lines, n_points), dtype=np.float32)
        read_idx = (_ls_head - _ls_n_valid + np.arange(_ls_n_valid)) % n_points
        logical[:, n_points - _ls_n_valid :] = _ls_buf[:, read_idx]
    else:
        read_idx = (_ls_head + np.arange(n_points)) % n_points
        logical = _ls_buf[:, read_idx]

    # push to each line graphic (this is the expensive part — N separate buffer updates)
    for i, line in enumerate(subplot["signals"]):
        line.data[:, 1] = logical[i]

    ls_times.append(time.perf_counter() - t0)
    if len(ls_times) >= 2:
        avg_us = np.mean(ls_times) * 1e6
        subplot.title = f"LineStack — {avg_us:.0f} \u00b5s/frame"

    # advance frame counter after both subplots have consumed the current value
    state["frame"] += 1


figure[0, 0].add_animations(update_multiline)
figure[0, 1].add_animations(update_linestack)

figure.show()

if __name__ == "__main__":
    if os.getenv("FPL_AUTOCLOSE"):
        fpl.loop.call_later(0.1, fpl.loop.stop)
    print(__doc__)
    fpl.loop.run()
