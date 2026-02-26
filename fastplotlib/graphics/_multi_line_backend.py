from __future__ import annotations

import numpy as np

from .features import MultiLinePositions


class MultiLineBackend:
    def __init__(self, data: MultiLinePositions, material):
        self._data = data
        self._material = material

    @property
    def n_lines(self) -> int:
        return self._data.n_lines

    @property
    def n_points(self) -> int:
        return self._data.n_points

    def set_scroll(self, head: int, n_valid: int | None = None) -> None:
        self._material.scroll_head = int(head)
        if n_valid is not None:
            self._material.scroll_n_valid = int(n_valid)

    def set_data(self, value) -> None:
        parsed = MultiLinePositions._parse_input(value)
        in_data, mode, n_lines, n_points = parsed

        if (n_lines, n_points) != (self.n_lines, self.n_points):
            raise ValueError("Full data assignment must match existing n_lines and n_points.")

        if mode in ("single_y", "multi_y"):
            x_values = np.arange(n_points, dtype=np.float32)
            use_shared_x = True
        else:
            xs = np.asarray(in_data[..., 0], dtype=np.float32)
            if np.allclose(xs, xs[0][None, :], equal_nan=False):
                x_values = xs[0]
                use_shared_x = True
            else:
                x_values = np.arange(n_points, dtype=np.float32)
                use_shared_x = False

        self._data[:] = value
        self._material.scroll_x_buffer.data[:] = x_values
        self._material.scroll_x_buffer.update_full()
        self._material.scroll_use_shared_x = use_shared_x

    def append_y(self, values: np.ndarray) -> None:
        if not self._material.scroll_enabled:
            raise RuntimeError("append_y is only available when scroll_enabled=True")

        y_new = np.asarray(values, dtype=np.float32)
        if y_new.ndim == 1:
            if y_new.shape[0] != self.n_lines:
                raise ValueError("append_y 1D values must have shape (n_lines,)")
            y_new = y_new[:, None]
        elif y_new.ndim == 2:
            if y_new.shape[0] != self.n_lines:
                raise ValueError("append_y 2D values must have shape (n_lines, n_new)")
        else:
            raise ValueError("append_y values must be 1D or 2D")

        n_new = int(y_new.shape[1])
        if n_new == 0:
            return

        n_points = self.n_points
        head = int(self._material.scroll_head)
        n_valid = int(self._material.scroll_n_valid)

        if n_new >= n_points:
            tail = y_new[:, -n_points:]
            self._data.set_y_point_ranges([(0, tail)])
            self._material.scroll_head = 0
            self._material.scroll_n_valid = n_points
            return

        y_updates: list[tuple[int, np.ndarray]] = []

        if n_valid < n_points:
            free = n_points - n_valid
            fill_count = min(free, n_new)
            if fill_count > 0:
                i0 = (head + n_valid) % n_points
                i1 = i0 + fill_count
                if i1 <= n_points:
                    y_updates.append((i0, y_new[:, :fill_count]))
                else:
                    split = n_points - i0
                    y_updates.append((i0, y_new[:, :split]))
                    y_updates.append((0, y_new[:, split:fill_count]))

                n_valid += fill_count
                y_new = y_new[:, fill_count:]
                n_new = y_new.shape[1]

        if n_new > 0:
            i0 = head
            i1 = head + n_new
            if i1 <= n_points:
                y_updates.append((i0, y_new))
            else:
                split = n_points - i0
                y_updates.append((i0, y_new[:, :split]))
                y_updates.append((0, y_new[:, split:]))

            head = (head + n_new) % n_points
            n_valid = n_points

        self._data.set_y_point_ranges(y_updates)
        self._material.scroll_head = head
        self._material.scroll_n_valid = n_valid
