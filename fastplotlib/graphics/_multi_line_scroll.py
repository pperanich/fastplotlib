from __future__ import annotations

import numpy as np
import pygfx
from pygfx.resources import Buffer
from pygfx.renderers.wgpu import Binding, register_wgpu_render_function
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


@register_wgpu_render_function(pygfx.Line, MultiLineScrollMaterial)
class MultiLineScrollShader(LineShader):
    def __init__(self, wobject):
        if not hasattr(LineShader, "get_line_access_helpers"):
            raise RuntimeError(
                "MultiLineScrollShader requires pygfx LineShader.get_line_access_helpers()."
            )

        super().__init__(wobject)

    def get_bindings(self, wobject, shared, scene=None):
        all_bindings = super().get_bindings(wobject, shared, scene)
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

    def get_line_access_helpers(self) -> str:
        color_helper = ""
        if self["color_mode"] == "vertex":
            color_type = {
                1: "f32",
                2: "vec2<f32>",
                3: "vec3<f32>",
                4: "vec4<f32>",
            }[self["color_buffer_channels"]]
            color_helper = f"""

fn gfx_load_node_color(i:i32) -> {color_type} {{
    return load_s_colors(gfx_map_node_index(i));
}}
"""

        texcoord_helper = ""
        if self["color_mode"] == "vertex_map":
            texcoord_type = {
                "1d": "f32",
                "2d": "vec2<f32>",
                "3d": "vec3<f32>",
            }[self["colormap_dim"]]
            texcoord_helper = f"""

fn gfx_load_node_texcoord(i:i32) -> {texcoord_type} {{
    return load_s_texcoords(gfx_map_node_index(i));
}}
"""

        return (
            """
fn multiline_col_from_index(i:i32) -> i32 {
    let stride = max(1, u_material.scroll_n_points + 1);
    let line_index = i / stride;
    return i - line_index * stride;
}

fn multiline_line_from_index(i:i32) -> i32 {
    let stride = max(1, u_material.scroll_n_points + 1);
    return i / stride;
}

fn gfx_map_node_index(i:i32) -> i32 {
    if (u_material.scroll_enabled == 0) {
        return i;
    }

    let n_points = max(1, u_material.scroll_n_points);
    let stride = n_points + 1;
    let line_index = i / stride;
    let col = i - line_index * stride;
    let base = line_index * stride;

    if (col >= n_points) {
        return base + n_points;
    }

    let n_valid = clamp(u_material.scroll_n_valid, 0, n_points);
    if (col < (n_points - n_valid)) {
        return base + n_points;
    }

    let mapped_col = (col + u_material.scroll_head + n_valid) % n_points;
    return base + mapped_col;
}

fn gfx_load_node_position(i:i32) -> vec3<f32> {
    let mapped_i = gfx_map_node_index(i);
    var pos_m = load_s_positions(mapped_i);

    let col_mapped = multiline_col_from_index(mapped_i);
    if (col_mapped >= u_material.scroll_n_points) {
        return pos_m;
    }

    let line_mapped = multiline_line_from_index(mapped_i);
    let y_i = col_mapped * u_material.scroll_n_lines + line_mapped;
    pos_m.y = load_s_scroll_y(y_i);

    if (u_material.scroll_use_shared_x != 0) {
        let col = multiline_col_from_index(i);
        if (col < u_material.scroll_n_points) {
            pos_m.x = load_s_scroll_x(col);
        }
    }

    return pos_m;
}
"""
            + color_helper
            + texcoord_helper
            + """

fn gfx_pick_node_index(i:i32) -> i32 {
    return gfx_map_node_index(i);
}
"""
        )
