"""
Multi-Line Single Buffer
========================

Plot many lines from a single buffer and update all y-values in one slice.
"""

import os

import numpy as np
import fastplotlib as fpl


n_lines = 64
n_points = 400

xs = np.linspace(0, 4 * np.pi, n_points, dtype=np.float32)
offsets = np.linspace(0, 20, n_lines, dtype=np.float32)

data = np.zeros((n_lines, n_points, 3), dtype=np.float32)
data[:, :, 0] = xs
data[:, :, 1] = np.sin(xs)[None, :]
data[:, :, 2] = offsets[:, None]

figure = fpl.Figure(size=(700, 560))
multi = figure[0, 0].add_multi_line(
    data,
    thickness=1.5,
    colors="w",
    name="signals",
    z_offset_scale=10.0,
)

phase = np.linspace(0, np.pi, n_lines, dtype=np.float32)[:, None]
state = {"t": 0.0}


def update_lines(subplot):
    state["t"] += 0.08
    subplot["signals"].data[:, :, 1] = np.sin(xs + phase + state["t"])


figure[0, 0].add_animations(update_lines)

figure[0, 0].axes.grids.xy.visible = True
figure.show()


# NOTE: fpl.loop.run() should not be used for interactive sessions
# See the "JupyterLab and IPython" section in the user guide
if __name__ == "__main__":
    if os.getenv("FPL_AUTOCLOSE"):
        fpl.loop.call_later(0.1, fpl.loop.stop)
    print(__doc__)
    fpl.loop.run()
