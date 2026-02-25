from typing import Any, Sequence
from warnings import warn

import numpy as np
import pygfx

from ...utils import (
    parse_cmap_values,
)
from ._base import (
    GraphicFeature,
    BufferManager,
    GraphicFeatureEvent,
    to_gpu_supported_dtype,
    block_reentrance,
)
from .utils import parse_colors


class VertexColors(BufferManager):
    event_info_spec = [
        {
            "dict key": "key",
            "type": "slice, index, numpy-like fancy index",
            "description": "index/slice at which colors were indexed/sliced",
        },
        {
            "dict key": "value",
            "type": "np.ndarray [n_points_changed, RGBA]",
            "description": "new color values for points that were changed",
        },
        {
            "dict key": "user_value",
            "type": "str or array-like",
            "description": "user input value that was parsed into the RGBA array",
        },
    ]

    def __init__(
        self,
        colors: str | pygfx.Color | np.ndarray | Sequence[float] | Sequence[str],
        n_colors: int,
        isolated_buffer: bool = True,
        property_name: str = "colors",
    ):
        """
        Manages the vertex color buffer for :class:`LineGraphic` or :class:`ScatterGraphic`

        Parameters
        ----------
        colors: str | pygfx.Color | np.ndarray | Sequence[float] | Sequence[str]
            specify colors as a single human-readable string, RGBA array,
            or an iterable of strings or RGBA arrays

        n_colors: int
            number of colors, if passing in a single str or single RGBA array

        """
        data = parse_colors(colors, n_colors)

        super().__init__(
            data=data, isolated_buffer=isolated_buffer, property_name=property_name
        )

    @block_reentrance
    def __setitem__(
        self,
        key: int | slice | np.ndarray[int | bool] | tuple[slice, ...],
        user_value: str | pygfx.Color | np.ndarray | Sequence[float] | Sequence[str],
    ):
        user_key = key

        if isinstance(key, tuple):
            # directly setting RGBA values for points, we do no parsing
            if not isinstance(user_value, (int, float, np.ndarray)):
                raise TypeError(
                    "Can only set from int, float, or array to set colors directly by slicing the entire array"
                )
            value = user_value

        elif isinstance(key, int):
            # set color of one point
            n_colors = 1
            value = parse_colors(user_value, n_colors)

        elif isinstance(key, slice):
            # find n_colors by converting slice to range and then parse colors
            start, stop, step = key.indices(self.value.shape[0])

            n_colors = len(range(start, stop, step))

            value = parse_colors(user_value, n_colors)

        elif isinstance(key, (np.ndarray, list)):
            if isinstance(key, list):
                # convert to array
                key = np.array(key)

            # make sure it's 1D
            if not key.ndim == 1:
                raise TypeError(
                    "If slicing colors with an array, it must be a 1D bool or int array"
                )

            if key.dtype == bool:
                # make sure len is same
                if not key.size == self.buffer.data.shape[0]:
                    raise IndexError(
                        f"Length of array for fancy indexing must match number of datapoints.\n"
                        f"There are {len(self.buffer.data.shape[0])} datapoints and you have passed {key.size} indices"
                    )
                n_colors = np.count_nonzero(key)

            elif np.issubdtype(key.dtype, np.integer):
                n_colors = key.size

            else:
                raise TypeError(
                    "If slicing colors with an array, it must be a 1D bool or int array"
                )

            value = parse_colors(user_value, n_colors)

        else:
            raise TypeError(
                f"invalid key for setting colors, you may set colors using integer indices, slices, or "
                f"fancy indexing using an array of integers or bool"
            )

        self.buffer.data[key] = value

        self._update_range(key)

        if len(self._event_handlers) < 1:
            return

        event_info = {
            "key": user_key,
            "value": value,
            "user_value": user_value,
        }

        event = GraphicFeatureEvent(self._property_name, info=event_info)
        self._call_event_handlers(event)

    def __len__(self):
        return len(self.buffer.data)


class UniformColor(GraphicFeature):
    event_info_spec = [
        {
            "dict key": "value",
            "type": "str | pygfx.Color | np.ndarray | Sequence[float]",
            "description": "new color value",
        },
    ]

    def __init__(
        self,
        value: str | pygfx.Color | np.ndarray | Sequence[float],
        property_name: str = "colors",
    ):
        """Manages uniform color for line or scatter material"""

        self._value = pygfx.Color(value)
        super().__init__(property_name=property_name)

    @property
    def value(self) -> pygfx.Color:
        return self._value

    @block_reentrance
    def set_value(
        self, graphic, value: str | pygfx.Color | np.ndarray | Sequence[float]
    ):
        value = pygfx.Color(value)
        graphic.world_object.material.color = value
        self._value = value

        event = GraphicFeatureEvent(type=self._property_name, info={"value": value})
        self._call_event_handlers(event)


class SizeSpace(GraphicFeature):
    event_info_spec = [
        {
            "dict key": "value",
            "type": "str",
            "description": "'screen' | 'world' | 'model'",
        },
    ]

    def __init__(self, value: str, property_name: str = "size_space"):
        """Manages the coordinate space for scatter/line graphic"""

        self._value = value
        super().__init__(property_name=property_name)

    @property
    def value(self) -> str:
        return self._value

    @block_reentrance
    def set_value(self, graphic, value: str):
        if value not in ["screen", "world", "model"]:
            raise ValueError(
                f"`size_space` must be one of: {['screen', 'world', 'model']}"
            )

        if "Line" in graphic.world_object.material.__class__.__name__:
            graphic.world_object.material.thickness_space = value
        else:
            graphic.world_object.material.size_space = value
        self._value = value

        event = GraphicFeatureEvent(type=self._property_name, info={"value": value})
        self._call_event_handlers(event)


class VertexPositions(BufferManager):
    event_info_spec = [
        {
            "dict key": "key",
            "type": "slice, index (int) or numpy-like fancy index",
            "description": "key at which vertex positions data were indexed/sliced",
        },
        {
            "dict key": "value",
            "type": "int | float | array-like",
            "description": "new data values for points that were changed",
        },
    ]

    def __init__(
        self, data: Any, isolated_buffer: bool = True, property_name: str = "data"
    ):
        """
        Manages the vertex positions buffer shown in the graphic.
        Supports fancy indexing if the data array also supports it.
        """

        data = self._fix_data(data)
        super().__init__(
            data, isolated_buffer=isolated_buffer, property_name=property_name
        )

    def _fix_data(self, data):
        if data.ndim == 1:
            # if user provides a 1D array, assume these are y-values
            data = np.column_stack([np.arange(data.size, dtype=data.dtype), data])

        if data.shape[1] != 3:
            if data.shape[1] != 2:
                raise ValueError(f"Must pass 1D, 2D or 3D data")

            # zeros for z
            zs = np.zeros(data.shape[0], dtype=data.dtype)

            # column stack [x, y, z] to make data of shape [n_points, 3]
            data = np.column_stack([data[:, 0], data[:, 1], zs])

        return to_gpu_supported_dtype(data)

    @block_reentrance
    def __setitem__(
        self,
        key: int | slice | np.ndarray[int | bool] | tuple[slice, ...],
        value: np.ndarray | float | list[float],
    ):
        # directly use the key to slice the buffer
        self.buffer.data[key] = value

        # _update_range handles parsing the key to
        # determine offset and size for GPU upload
        self._update_range(key)

        self._emit_event(self._property_name, key, value)

    def __len__(self):
        return len(self.buffer.data)


class MultiLinePositions(BufferManager):
    event_info_spec = [
        {
            "dict key": "key",
            "type": "slice, index (int) or numpy-like fancy index",
            "description": "key at which multi-line positions data were indexed/sliced",
        },
        {
            "dict key": "value",
            "type": "int | float | array-like",
            "description": "new data values for points that were changed",
        },
    ]

    def __init__(
        self, data: Any, isolated_buffer: bool = True, property_name: str = "data"
    ):
        parsed = self._parse_input(data)
        in_data, mode, n_lines, n_points = parsed
        self._n_lines = n_lines
        self._n_points = n_points

        out_dtype = in_data.dtype
        if out_dtype != np.float32:
            warn(f"casting {out_dtype} array to float32")
            out_dtype = np.float32

        packed = np.empty((self._n_lines, self._n_points + 1, 3), dtype=out_dtype)
        self._fill_packed(packed[:, :-1, :], in_data, mode)
        packed[:, self._n_points, :] = np.nan

        flat = packed.reshape(-1, 3)
        super().__init__(
            flat, isolated_buffer=isolated_buffer, property_name=property_name
        )

        self._data_3d = self._buffer.data.reshape(self._n_lines, self._n_points + 1, 3)
        self._data_view = self._data_3d[:, :-1, :]
        self._data_3d[:, -1, :] = np.nan
        self._x_values_shared = None
        self._update_x_values_shared()

    def _update_x_values_shared(self) -> None:
        xs = self._data_view[:, :, 0]
        if np.allclose(xs, xs[0][None, :], equal_nan=False):
            self._x_values_shared = xs[0].copy()
        else:
            self._x_values_shared = None

    @staticmethod
    def _parse_input(data):
        data = np.asarray(data)

        if data.dtype == object:
            raise ValueError(
                "MultiLine data must be a rectangular array with the same number of points per line"
            )

        if data.ndim == 1:
            return data, "single_y", 1, data.shape[0]

        if data.ndim == 2:
            return data, "multi_y", data.shape[0], data.shape[1]

        if data.ndim != 3:
            raise ValueError("MultiLine data must be 1D, 2D, or 3D")

        if data.shape[2] not in (2, 3):
            raise ValueError("MultiLine data last dimension must be 2 or 3")

        mode = "multi_xyz" if data.shape[2] == 3 else "multi_xy"
        return data, mode, data.shape[0], data.shape[1]

    def _fill_packed(self, packed, in_data, mode):
        if mode == "single_y":
            n_points = packed.shape[1]
            packed[0, :, 0] = np.arange(n_points, dtype=packed.dtype)
            packed[0, :, 1] = in_data
            packed[0, :, 2] = 0.0
            return

        if mode == "multi_y":
            n_points = packed.shape[1]
            packed[:, :, 0] = np.arange(n_points, dtype=packed.dtype)[None, :]
            packed[:, :, 1] = in_data
            packed[:, :, 2] = 0.0
            return

        if mode in ("single_xy", "multi_xy"):
            packed[:, :, 0] = in_data[..., 0]
            packed[:, :, 1] = in_data[..., 1]
            packed[:, :, 2] = 0.0
            return

        if mode in ("single_xyz", "multi_xyz"):
            packed[:, :, :] = in_data
            return

        raise ValueError(f"Unknown MultiLine input mode: {mode}")

    @property
    def value(self) -> np.ndarray:
        return self._data_view

    @property
    def n_lines(self) -> int:
        return self._n_lines

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def flat_value(self) -> np.ndarray:
        return self._buffer.data

    @property
    def x_values_shared(self) -> np.ndarray | None:
        """Shared x-values used by all lines, if available."""
        return self._x_values_shared

    @block_reentrance
    def __setitem__(
        self,
        key: int | slice | np.ndarray[int | bool] | tuple[slice, ...],
        value: np.ndarray | float | list[float],
    ):
        if self._is_full_slice(key):
            parsed = self._parse_input(value)
            in_data, mode, n_lines, n_points = parsed

            if (n_lines, n_points) != (self._n_lines, self._n_points):
                raise ValueError(
                    "Full data assignment must match existing n_lines and n_points."
                )

            self._fill_packed(self._data_3d[:, :-1, :], in_data, mode)
            self._data_3d[:, -1, :] = np.nan
            self._update_x_values_shared()
        else:
            self._data_view[key] = value

        self.buffer.update_full()
        self._emit_event(self._property_name, key, value)

    def __getitem__(self, item):
        return self._data_view[item]

    def __len__(self):
        return self._n_lines

    def mark_point_ranges_dirty(self, segments: list[tuple[int, int]]):
        """
        Mark point segments as dirty for GPU upload.

        Parameters
        ----------
        segments:
            List of ``(start, count)`` point ranges in ``[0, n_points)``.
            These are applied for every line in the packed multiline buffer.
        """
        if not segments:
            return

        stride = self._n_points + 1
        n_points = self._n_points

        for start, count in segments:
            if count <= 0:
                continue
            if start < 0 or start >= n_points:
                raise ValueError("point start out of range for MultiLinePositions")
            if start + count > n_points:
                raise ValueError("point range exceeds n_points in MultiLinePositions")

            for line_i in range(self._n_lines):
                offset = line_i * stride + start
                self.buffer.update_range(offset=offset, size=count)

    def update_columns(self, segments: list[tuple[int, int]]):
        # Backwards compat for earlier internal naming; use point terminology.
        self.mark_point_ranges_dirty(segments)

    @staticmethod
    def _is_full_slice(key) -> bool:
        if key == slice(None):
            return True

        if not isinstance(key, tuple):
            return False

        if any(k is Ellipsis for k in key):
            return False

        return all(k == slice(None) for k in key)


class VertexCmap(BufferManager):
    event_info_spec = [
        {
            "dict key": "key",
            "type": "slice",
            "description": "key at cmap colors were sliced",
        },
        {
            "dict key": "value",
            "type": "str",
            "description": "new cmap to set at given slice",
        },
    ]

    def __init__(
        self,
        vertex_colors: VertexColors,
        cmap_name: str | None,
        transform: np.ndarray | None,
        property_name: str = "colors",
    ):
        """
        Sliceable colormap feature, manages a VertexColors instance and
        provides a way to set colormaps with arbitrary transforms
        """

        super().__init__(data=vertex_colors.buffer, property_name=property_name)

        self._vertex_colors = vertex_colors
        self._cmap_name = cmap_name
        self._transform = transform

        if self._cmap_name is not None:
            if not isinstance(self._cmap_name, str):
                raise TypeError(
                    f"cmap name must be of type <str>, you have passed: {self._cmap_name} of type: {type(self._cmap_name)}"
                )

            if self._transform is not None:
                self._transform = np.asarray(self._transform)

            n_datapoints = vertex_colors.value.shape[0]

            colors = parse_cmap_values(
                n_colors=n_datapoints,
                cmap_name=self._cmap_name,
                transform=self._transform,
            )
            # set vertex colors from cmap
            self._vertex_colors[:] = colors

    @block_reentrance
    def __setitem__(self, key: slice, cmap_name):
        if not isinstance(key, slice):
            raise TypeError(
                "fancy indexing not supported for VertexCmap, only slices "
                "of a continuous range are supported for applying a cmap"
            )
        if key.step is not None:
            raise TypeError(
                "step sized indexing not currently supported for setting VertexCmap, "
                "slices must be a continuous range"
            )

        # parse slice
        start, stop, step = key.indices(self.value.shape[0])
        n_elements = len(range(start, stop, step))

        colors = parse_cmap_values(
            n_colors=n_elements, cmap_name=cmap_name, transform=self._transform
        )

        self._cmap_name = cmap_name
        self._vertex_colors[key] = colors

        # TODO: should we block vertex_colors from emitting an event?
        #  Because currently this will result in 2 emitted events, one
        #  for cmap and another from the colors
        self._emit_event(self._property_name, key, cmap_name)

    @property
    def name(self) -> str:
        return self._cmap_name

    @property
    def transform(self) -> np.ndarray | None:
        """Get or set the cmap transform. Maps values from the transform array to the cmap colors"""
        return self._transform

    @transform.setter
    def transform(
        self,
        values: np.ndarray | list[float | int],
        indices: slice | list | np.ndarray = None,
    ):
        if self._cmap_name is None:
            raise AttributeError(
                "cmap name is not set, set the cmap name before setting the transform"
            )

        values = np.asarray(values)

        colors = parse_cmap_values(
            n_colors=self.value.shape[0], cmap_name=self._cmap_name, transform=values
        )

        self._transform = values

        if indices is None:
            indices = slice(None)

        self._vertex_colors[indices] = colors

        self._emit_event("cmap.transform", indices, values)

    def __len__(self):
        raise NotImplementedError(
            "len not implemented for `cmap`, use len(colors) instead"
        )

    def __repr__(self):
        return f"{self.__class__.__name__} | cmap: {self.name}\ntransform: {self.transform}"
