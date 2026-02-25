"""
Multi-Line Ring Scroll
======================

Scroll many lines by appending only new y-values to a ring buffer.
"""

import os

import numpy as np

import fastplotlib as fpl


n_lines = 64
n_points = 80000

# Preallocate y-only data; x is generated [0..n_points-1].
data = np.zeros((n_lines, n_points), dtype=np.float32)
phase = np.linspace(0, np.pi, n_lines, dtype=np.float32)[:, None]
offsets = np.linspace(0.0, 18.0, n_lines, dtype=np.float32)[:, None]

figure = fpl.Figure(size=(900, 560))
multi = figure[0, 0].add_multi_line(
    data,
    thickness=1.2,
    colors="w",
    name="signals",
    scroll=True,
    scroll_n_valid=0,
)

state = {"t": 0.0}


def update_lines(subplot):
    k = 8
    t = state["t"] + 0.06 * np.arange(1, k + 1, dtype=np.float32)
    y = np.sin(t[None, :] + phase) + offsets
    subplot["signals"].append_y(y)
    state["t"] = float(t[-1])


figure[0, 0].add_animations(update_lines)
figure[0, 0].axes.grids.xy.visible = True
figure.show()

if __name__ == "__main__":
    if os.getenv("FPL_AUTOCLOSE"):
        fpl.loop.call_later(0.1, fpl.loop.stop)
    print(__doc__)
    fpl.loop.run()
