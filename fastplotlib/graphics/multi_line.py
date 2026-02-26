from typing import Any, Sequence
from types import SimpleNamespace

import numpy as np
import pygfx

from ._base import Graphic
from ._multi_line_scroll import MultiLineScrollMaterial
from .features import (
    Thickness,
    VertexColors,
    UniformColor,
    VertexCmap,
    SizeSpace,
    MultiLinePositions,
)


class MultiLineGraphic(Graphic):
    _features = {
        "data": MultiLinePositions,
        "colors": (VertexColors, UniformColor),
        "cmap": (VertexCmap, None),  # none if UniformColor
        "thickness": Thickness,
        "size_space": SizeSpace,
    }

    def __init__(
        self,
        data: Any,
        thickness: float = 2.0,
        colors: str | np.ndarray | Sequence = "w",
        uniform_color: bool = False,
        cmap: str = None,
        cmap_transform: np.ndarray | Sequence = None,
        z_offset_scale: float | None = None,
        isolated_buffer: bool = True,
        size_space: str = "screen",
        scroll: bool = False,
        scroll_head: int = 0,
        scroll_n_valid: int | None = None,
        **kwargs,
    ):
        """
        Create a multi-line Graphic backed by a single positions buffer.

        Parameters
        ----------
        data: array-like
            Multi-line data to plot. Accepts:
            - 1D array for a single line (y-values, x generated)
            - 2D array [n_lines, n_points] for y-values (x generated)
            - 3D array [n_lines, n_points, 2|3] for explicit x/y[/z]

        thickness: float, optional, default 2.0
            thickness of the line

        colors: str, array, or iterable, default "w"
            specify colors as a single human-readable string, a single RGBA array,
            or a Sequence (array, tuple, or list) of strings or RGBA arrays

        uniform_color: bool, default ``False``
            if True, uses a uniform buffer for the line color

        cmap: str, optional
            Apply a colormap to the line instead of assigning colors manually

        cmap_transform: 1D array-like of numerical values, optional
            if provided, these values are used to map the colors from the cmap

        z_offset_scale: float, optional
            if provided, applies a shear transform so that y' = y + z * z_offset_scale.
            This treats the z coordinate as a per-vertex y-offset and is intended for 2D use.

        size_space: str, default "screen"
            coordinate space in which the thickness is expressed ("screen", "world", "model")

        scroll: bool, default False
            Enable ring-buffered scrolling mode. Rendering uses a shader remap
            controlled by ``scroll_head`` / ``scroll_n_valid``.

        scroll_head: int, default 0
            Index in the ring buffer representing the current logical origin.

        scroll_n_valid: int, optional
            Number of valid samples in the ring buffer. If omitted, defaults to
            full capacity (``n_points``).

        **kwargs
            passed to :class:`.Graphic`
        """

        self._data = MultiLinePositions(data, isolated_buffer=isolated_buffer)

        n_vertices = self._data.flat_value.shape[0]
        n_lines = self._data.n_lines
        n_points = self._data.n_points

        if cmap_transform is not None and cmap is None:
            raise ValueError("must pass `cmap` if passing `cmap_transform`")

        if cmap is not None:
            if uniform_color:
                raise TypeError("Cannot use cmap if uniform_color=True")

            if isinstance(cmap, str):
                if isinstance(colors, VertexColors):
                    self._colors = colors
                    self._colors._shared += 1
                else:
                    colors = self._expand_line_colors(colors, n_lines, n_points)
                    self._colors = VertexColors(
                        colors, n_colors=n_vertices, isolated_buffer=isolated_buffer
                    )

                self._cmap = VertexCmap(
                    self._colors,
                    cmap_name=cmap,
                    transform=cmap_transform,
                )
            elif isinstance(cmap, VertexCmap):
                self._cmap = cmap
                self._colors = cmap._vertex_colors
            else:
                raise TypeError(
                    "`cmap` argument must be a <str> cmap name or an existing `VertexCmap` instance"
                )
        else:
            if isinstance(colors, VertexColors):
                self._colors = colors
                self._colors._shared += 1
                self._cmap = VertexCmap(self._colors, cmap_name=None, transform=None)
            else:
                if uniform_color:
                    if not isinstance(colors, str):
                        if not len(colors) in [3, 4]:
                            raise TypeError(
                                "must pass a single color if using `uniform_colors=True`"
                            )
                    self._colors = UniformColor(colors)
                    self._cmap = None
                else:
                    colors = self._expand_line_colors(colors, n_lines, n_points)
                    self._colors = VertexColors(
                        colors, n_colors=n_vertices, isolated_buffer=isolated_buffer
                    )
                    self._cmap = VertexCmap(
                        self._colors, cmap_name=None, transform=None
                    )

        self._size_space = SizeSpace(size_space)
        super().__init__(**kwargs)

        self._thickness = Thickness(thickness)

        MaterialCls = MultiLineScrollMaterial
        aa = kwargs.get("alpha_mode", "auto") in ("blend", "weighted_blend")
        x_values_shared = self._data.x_values_shared
        if x_values_shared is None:
            x_values_shared = np.arange(n_points, dtype=np.float32)
            use_shared_x = False
        else:
            use_shared_x = True

        if uniform_color:
            geometry = pygfx.Geometry(positions=self._data.buffer)
            material_kwargs = dict(
                aa=aa,
                thickness=self.thickness,
                color_mode="uniform",
                color=self.colors,
                pick_write=True,
                thickness_space=self.size_space,
                depth_compare="<=",
            )
            material = MaterialCls(
                scroll_x=x_values_shared,
                scroll_y=self._data.y_buffer,
                scroll_n_points=n_points,
                scroll_n_lines=n_lines,
                scroll_enabled=scroll,
                scroll_head=scroll_head,
                scroll_n_valid=scroll_n_valid,
                scroll_use_shared_x=use_shared_x,
                **material_kwargs,
            )
        else:
            material_kwargs = dict(
                aa=aa,
                thickness=self.thickness,
                color_mode="vertex",
                pick_write=True,
                thickness_space=self.size_space,
                depth_compare="<=",
            )
            material = MaterialCls(
                scroll_x=x_values_shared,
                scroll_y=self._data.y_buffer,
                scroll_n_points=n_points,
                scroll_n_lines=n_lines,
                scroll_enabled=scroll,
                scroll_head=scroll_head,
                scroll_n_valid=scroll_n_valid,
                scroll_use_shared_x=use_shared_x,
                **material_kwargs,
            )
            geometry = pygfx.Geometry(
                positions=self._data.buffer, colors=self._colors.buffer
            )

        line_world_object: pygfx.Line = pygfx.Line(geometry=geometry, material=material)
        self._line_world_object = line_world_object
        self._feature_target = SimpleNamespace(world_object=self._line_world_object)

        self._z_offset_scale = z_offset_scale
        if z_offset_scale not in (None, 0.0):
            self._apply_z_offset_shear(z_offset_scale)

        world_object = pygfx.Group()
        world_object.add(line_world_object)
        self._set_world_object(world_object)
        self._line_world_object.material.opacity = self.alpha
        self._line_world_object.material.alpha_mode = self.alpha_mode

    @property
    def scroll_enabled(self) -> bool:
        material = self._line_world_object.material
        if not isinstance(material, MultiLineScrollMaterial):
            return False
        return material.scroll_enabled

    @property
    def scroll_head(self) -> int:
        if not self.scroll_enabled:
            return 0
        return self._line_world_object.material.scroll_head

    @property
    def scroll_n_valid(self) -> int:
        if not self.scroll_enabled:
            return self.n_points
        return self._line_world_object.material.scroll_n_valid

    def set_scroll_head(self, head: int, n_valid: int | None = None):
        if not self.scroll_enabled:
            raise RuntimeError("scroll mode is disabled for this MultiLineGraphic")

        material: MultiLineScrollMaterial = self._line_world_object.material
        material.scroll_head = int(head)
        if n_valid is not None:
            material.scroll_n_valid = int(n_valid)

    def append_y(self, values: np.ndarray):
        """
        Append y-values into the ring buffer and advance scroll state.

        Parameters
        ----------
        values: np.ndarray
            Shape ``(n_lines,)`` for one sample, or ``(n_lines, n_new)`` for a batch.
        """
        if not self.scroll_enabled:
            raise RuntimeError("append_y is only available when scroll=True")

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

        material: MultiLineScrollMaterial = self._line_world_object.material
        n_points = self.n_points
        head = material.scroll_head
        n_valid = material.scroll_n_valid
        y_data = self.data.value[:, :, 1]
        dirty_point_segments: list[tuple[int, int]] = []

        def add_segment(start: int, count: int):
            if count <= 0:
                return
            start = int(start) % n_points
            end = start + count
            if end <= n_points:
                dirty_point_segments.append((start, count))
            else:
                split = n_points - start
                dirty_point_segments.append((start, split))
                dirty_point_segments.append((0, end % n_points))

        if n_new >= n_points:
            tail = y_new[:, -n_points:]
            y_data[:, :] = tail
            material.scroll_head = 0
            material.scroll_n_valid = n_points
            self.data.mark_y_point_ranges_dirty([(0, n_points)])
            return

        if n_valid < n_points:
            free = n_points - n_valid
            fill_count = min(free, n_new)
            if fill_count > 0:
                i0 = (head + n_valid) % n_points
                i1 = i0 + fill_count
                if i1 <= n_points:
                    y_data[:, i0:i1] = y_new[:, :fill_count]
                else:
                    split = n_points - i0
                    y_data[:, i0:] = y_new[:, :split]
                    y_data[:, : i1 % n_points] = y_new[:, split:fill_count]
                add_segment(i0, fill_count)
                n_valid += fill_count
                y_new = y_new[:, fill_count:]
                n_new = y_new.shape[1]

        if n_new > 0:
            i0 = head
            i1 = head + n_new
            if i1 <= n_points:
                y_data[:, i0:i1] = y_new
            else:
                split = n_points - i0
                y_data[:, i0:] = y_new[:, :split]
                y_data[:, : i1 % n_points] = y_new[:, split:]
            add_segment(i0, n_new)
            head = (head + n_new) % n_points
            n_valid = n_points

        material.scroll_head = head
        material.scroll_n_valid = n_valid
        self.data.mark_y_point_ranges_dirty(dirty_point_segments)

    def _apply_z_offset_shear(self, scale: float):
        shear = np.eye(4, dtype=np.float32)
        shear[1, 2] = float(scale)
        self._line_world_object.local.state_basis = "matrix"
        self._line_world_object.local.matrix = (
            shear @ self._line_world_object.local.matrix
        )

    @staticmethod
    def _expand_line_colors(colors, n_lines: int, n_points: int):
        if isinstance(colors, np.ndarray):
            if (
                colors.ndim == 2
                and colors.shape[0] == n_lines
                and colors.shape[1]
                in (
                    3,
                    4,
                )
            ):
                line_colors = colors
            else:
                return colors
        elif isinstance(colors, (list, tuple)) and len(colors) == n_lines:
            if len(colors) == 4 and all(isinstance(c, (float, int)) for c in colors):
                return colors
            line_colors = np.array([pygfx.Color(c) for c in colors])
        else:
            return colors

        if line_colors.shape[1] == 3:
            alpha = np.ones((n_lines, 1), dtype=line_colors.dtype)
            line_colors = np.concatenate([line_colors, alpha], axis=1)

        expanded = np.repeat(line_colors[:, None, :], n_points + 1, axis=1)
        return expanded.reshape(-1, 4)

    @property
    def data(self) -> MultiLinePositions:
        """Get or set the graphic's data"""
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(self._line_world_object.material, MultiLineScrollMaterial):
            parsed = MultiLinePositions._parse_input(value)
            in_data, mode, n_lines, n_points = parsed

            if (n_lines, n_points) != (self.n_lines, self.n_points):
                raise ValueError(
                    "Full data assignment must match existing n_lines and n_points."
                )

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
            material: MultiLineScrollMaterial = self._line_world_object.material
            material.scroll_x_buffer.data[:] = x_values
            material.scroll_x_buffer.update_full()
            material.scroll_use_shared_x = use_shared_x
            return

        self._data[:] = value

    @property
    def colors(self) -> VertexColors | pygfx.Color:
        """Get or set the colors"""
        if isinstance(self._colors, VertexColors):
            return self._colors

        if isinstance(self._colors, UniformColor):
            return self._colors.value

        raise RuntimeError("Unknown colors type on MultiLineGraphic")

    @colors.setter
    def colors(self, value: str | np.ndarray | Sequence[float] | Sequence[str]):
        if isinstance(self._colors, VertexColors):
            self._colors[:] = value
            return

        if isinstance(self._colors, UniformColor):
            self._colors.set_value(self._feature_target, value)
            return

        raise RuntimeError("Unknown colors type on MultiLineGraphic")

    @property
    def cmap(self) -> VertexCmap:
        """
        Control the cmap or cmap transform
        """
        return self._cmap

    @cmap.setter
    def cmap(self, name: str):
        if self._cmap is None:
            raise BufferError("Cannot use cmap with uniform_colors=True")

        self._cmap[:] = name

    @property
    def size_space(self):
        """
        The coordinate space in which the size is expressed ('screen', 'world', 'model')
        """
        return self._size_space.value

    @size_space.setter
    def size_space(self, value: str):
        self._size_space.set_value(self._feature_target, value)

    @property
    def thickness(self) -> float:
        """Get or set the line thickness"""
        return self._thickness.value

    @thickness.setter
    def thickness(self, value: float):
        self._thickness.set_value(self._feature_target, value)

    @property
    def alpha(self) -> float:
        return self._alpha.value

    @alpha.setter
    def alpha(self, value: float):
        self._alpha.set_value(self._feature_target, value)

    @property
    def alpha_mode(self) -> str:
        return self._alpha_mode.value

    @alpha_mode.setter
    def alpha_mode(self, value: str):
        self._alpha_mode.set_value(self._feature_target, value)

    @property
    def n_lines(self) -> int:
        return self._data.n_lines

    @property
    def n_points(self) -> int:
        return self._data.n_points

    def format_pick_info(self, pick_info: dict) -> str:
        index = pick_info["vertex_index"]
        data = self._data.flat_value[index]
        info = "\n".join(f"{dim}: {val:.4g}" for dim, val in zip("xyz", data))
        return info
