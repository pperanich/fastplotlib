from typing import Any, Sequence

import numpy as np
import pygfx

from ._base import Graphic
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

        MaterialCls = pygfx.LineMaterial
        aa = kwargs.get("alpha_mode", "auto") in ("blend", "weighted_blend")

        if uniform_color:
            geometry = pygfx.Geometry(positions=self._data.buffer)
            material = MaterialCls(
                aa=aa,
                thickness=self.thickness,
                color_mode="uniform",
                color=self.colors,
                pick_write=True,
                thickness_space=self.size_space,
                depth_compare="<=",
            )
        else:
            material = MaterialCls(
                aa=aa,
                thickness=self.thickness,
                color_mode="vertex",
                pick_write=True,
                thickness_space=self.size_space,
                depth_compare="<=",
            )
            geometry = pygfx.Geometry(
                positions=self._data.buffer, colors=self._colors.buffer
            )

        world_object: pygfx.Line = pygfx.Line(geometry=geometry, material=material)
        self._set_world_object(world_object)

        self._z_offset_scale = z_offset_scale
        if z_offset_scale not in (None, 0.0):
            self._apply_z_offset_shear(z_offset_scale)

    def _apply_z_offset_shear(self, scale: float):
        shear = np.eye(4, dtype=np.float32)
        shear[1, 2] = float(scale)
        self.transform = shear @ self.world_object.local.matrix

    @staticmethod
    def _expand_line_colors(colors, n_lines: int, n_points: int):
        if isinstance(colors, np.ndarray):
            if colors.ndim == 2 and colors.shape[0] == n_lines and colors.shape[1] in (
                3,
                4,
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
            self._colors.set_value(self, value)
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
        self._size_space.set_value(self, value)

    @property
    def thickness(self) -> float:
        """Get or set the line thickness"""
        return self._thickness.value

    @thickness.setter
    def thickness(self, value: float):
        self._thickness.set_value(self, value)

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
