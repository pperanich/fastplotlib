from __future__ import annotations

import numpy as np
import pygfx
from importlib import resources
from pygfx.resources import Buffer
from pygfx.renderers.wgpu import Binding, load_wgsl, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


class MultiLineScrollMaterial(pygfx.LineMaterial):
    """Line material with uniforms/buffers for ring-buffered multiline scrolling."""

    uniform_type = dict(
        pygfx.LineMaterial.uniform_type,
        scroll_enabled="i4",
        scroll_head="i4",
        scroll_n_points="i4",
        scroll_n_lines="i4",
        scroll_n_valid="i4",
        scroll_use_shared_x="i4",
    )

    def __init__(
        self,
        *,
        scroll_x: np.ndarray,
        scroll_y: np.ndarray | Buffer,
        scroll_n_points: int,
        scroll_n_lines: int,
        scroll_enabled: bool = True,
        scroll_head: int = 0,
        scroll_n_valid: int | None = None,
        scroll_use_shared_x: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        x = np.asarray(scroll_x, dtype=np.float32).reshape(-1)
        if x.size != int(scroll_n_points):
            raise ValueError(
                "scroll_x length must match scroll_n_points for MultiLine scrolling"
            )

        self._scroll_x_buffer = Buffer(x)
        if isinstance(scroll_y, Buffer):
            self._scroll_y_buffer = scroll_y
        else:
            y = np.asarray(scroll_y, dtype=np.float32).reshape(-1)
            if y.size != int(scroll_n_points) * int(scroll_n_lines):
                raise ValueError(
                    "scroll_y length must match scroll_n_points * scroll_n_lines"
                )
            self._scroll_y_buffer = Buffer(y)
        self.scroll_n_points = int(scroll_n_points)
        self.scroll_n_lines = int(scroll_n_lines)
        self.scroll_head = int(scroll_head)
        self.scroll_n_valid = (
            int(scroll_n_points) if scroll_n_valid is None else int(scroll_n_valid)
        )
        self.scroll_use_shared_x = bool(scroll_use_shared_x)
        self.scroll_enabled = bool(scroll_enabled)

    @property
    def scroll_x_buffer(self) -> Buffer:
        return self._scroll_x_buffer

    @property
    def scroll_y_buffer(self) -> Buffer:
        return self._scroll_y_buffer

    @property
    def scroll_enabled(self) -> bool:
        return bool(self.uniform_buffer.data["scroll_enabled"])

    @scroll_enabled.setter
    def scroll_enabled(self, value: bool):
        self.uniform_buffer.data["scroll_enabled"] = 1 if bool(value) else 0
        self.uniform_buffer.update_full()

    @property
    def scroll_head(self) -> int:
        return int(self.uniform_buffer.data["scroll_head"])

    @scroll_head.setter
    def scroll_head(self, value: int):
        n_points = max(1, self.scroll_n_points)
        self.uniform_buffer.data["scroll_head"] = int(value) % n_points
        self.uniform_buffer.update_full()

    @property
    def scroll_n_points(self) -> int:
        return int(self.uniform_buffer.data["scroll_n_points"])

    @scroll_n_points.setter
    def scroll_n_points(self, value: int):
        self.uniform_buffer.data["scroll_n_points"] = max(1, int(value))
        self.uniform_buffer.update_full()

    @property
    def scroll_n_valid(self) -> int:
        return int(self.uniform_buffer.data["scroll_n_valid"])

    @scroll_n_valid.setter
    def scroll_n_valid(self, value: int):
        n_points = self.scroll_n_points
        self.uniform_buffer.data["scroll_n_valid"] = max(0, min(int(value), n_points))
        self.uniform_buffer.update_full()

    @property
    def scroll_n_lines(self) -> int:
        return int(self.uniform_buffer.data["scroll_n_lines"])

    @scroll_n_lines.setter
    def scroll_n_lines(self, value: int):
        self.uniform_buffer.data["scroll_n_lines"] = max(1, int(value))
        self.uniform_buffer.update_full()

    @property
    def scroll_use_shared_x(self) -> bool:
        return bool(self.uniform_buffer.data["scroll_use_shared_x"])

    @scroll_use_shared_x.setter
    def scroll_use_shared_x(self, value: bool):
        self.uniform_buffer.data["scroll_use_shared_x"] = 1 if bool(value) else 0
        self.uniform_buffer.update_full()


def _load_multiline_line_wgsl() -> str:
    data = resources.files("fastplotlib.graphics.wgsl").joinpath(
        "multi_line_line.wgsl"
    ).read_bytes()
    return data.decode("utf-8")


_SCROLL_LINE_WGSL = _load_multiline_line_wgsl()


@register_wgpu_render_function(pygfx.Line, MultiLineScrollMaterial)
class MultiLineScrollShader(LineShader):
    def get_bindings(self, wobject, shared):
        all_bindings = super().get_bindings(wobject, shared)
        bindings0 = dict(all_bindings[0])
        binding_index = max(bindings0.keys()) + 1 if bindings0 else 0
        bindings0[binding_index] = Binding(
            "s_scroll_x",
            "buffer/read_only_storage",
            wobject.material.scroll_x_buffer,
            "VERTEX",
        )
        bindings0[binding_index + 1] = Binding(
            "s_scroll_y",
            "buffer/read_only_storage",
            wobject.material.scroll_y_buffer,
            "VERTEX",
        )
        self.define_bindings(0, bindings0)
        all_bindings[0] = bindings0
        return all_bindings

    def get_code(self):
        return _SCROLL_LINE_WGSL
