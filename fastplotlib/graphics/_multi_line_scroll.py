from __future__ import annotations

import numpy as np
import pygfx
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


def _build_scroll_line_wgsl() -> str:
    code = load_wgsl("line.wgsl")

    marker = "// -------------------- vertex shader --------------------"
    if marker not in code:
        raise RuntimeError(
            "Failed to patch pygfx line shader: missing vertex shader marker"
        )

    helper_block = """
fn multiline_col_from_index(i:i32) -> i32 {
    let stride = max(1, u_material.scroll_n_points + 1);
    let line_index = i / stride;
    return i - line_index * stride;
}

fn multiline_line_from_index(i:i32) -> i32 {
    let stride = max(1, u_material.scroll_n_points + 1);
    return i / stride;
}

fn remap_multiline_index(i:i32) -> i32 {
    if (u_material.scroll_enabled == 0) {
        return i;
    }

    let n_points = max(1, u_material.scroll_n_points);
    let stride = n_points + 1;
    let line_index = i / stride;
    let col = i - line_index * stride;
    let base = line_index * stride;

    if (col >= n_points) {
        return base + n_points;  // preserve separator
    }

    let n_valid = clamp(u_material.scroll_n_valid, 0, n_points);
    if (col < (n_points - n_valid)) {
        // Left/startup blank region: map to separator (NaN vertex)
        return base + n_points;
    }

    // For partial fill, right-aligned visible region starts at (n_points - n_valid).
    // This reduces to (col + scroll_head) when n_valid == n_points.
    let mapped_col = (col + u_material.scroll_head + n_valid) % n_points;
    return base + mapped_col;
}
"""
    code = code.replace(marker, helper_block + "\n" + marker)

    old_positions = """    var pos_m_prev = load_s_positions(node_index_prev);
    var pos_m_node = load_s_positions(node_index);
    var pos_m_next = load_s_positions(node_index_next);"""
    new_positions = """    let node_index_prev_mapped = remap_multiline_index(node_index_prev);
    let node_index_mapped = remap_multiline_index(node_index);
    let node_index_next_mapped = remap_multiline_index(node_index_next);

    var pos_m_prev = load_s_positions(node_index_prev_mapped);
    var pos_m_node = load_s_positions(node_index_mapped);
    var pos_m_next = load_s_positions(node_index_next_mapped);

    let col_prev = multiline_col_from_index(node_index_prev);
    let col_node = multiline_col_from_index(node_index);
    let col_next = multiline_col_from_index(node_index_next);

    let col_prev_mapped = multiline_col_from_index(node_index_prev_mapped);
    let col_node_mapped = multiline_col_from_index(node_index_mapped);
    let col_next_mapped = multiline_col_from_index(node_index_next_mapped);

    let line_prev_mapped = multiline_line_from_index(node_index_prev_mapped);
    let line_node_mapped = multiline_line_from_index(node_index_mapped);
    let line_next_mapped = multiline_line_from_index(node_index_next_mapped);

    if (col_prev_mapped < u_material.scroll_n_points) {
        let y_i = col_prev_mapped * u_material.scroll_n_lines + line_prev_mapped;
        pos_m_prev.y = load_s_scroll_y(y_i);
    }
    if (col_node_mapped < u_material.scroll_n_points) {
        let y_i = col_node_mapped * u_material.scroll_n_lines + line_node_mapped;
        pos_m_node.y = load_s_scroll_y(y_i);
    }
    if (col_next_mapped < u_material.scroll_n_points) {
        let y_i = col_next_mapped * u_material.scroll_n_lines + line_next_mapped;
        pos_m_next.y = load_s_scroll_y(y_i);
    }

    if (u_material.scroll_use_shared_x != 0) {
        if (col_prev < u_material.scroll_n_points) {
            pos_m_prev.x = load_s_scroll_x(col_prev);
        }
        if (col_node < u_material.scroll_n_points) {
            pos_m_node.x = load_s_scroll_x(col_node);
        }
        if (col_next < u_material.scroll_n_points) {
            pos_m_next.x = load_s_scroll_x(col_next);
        }
    }"""
    if old_positions not in code:
        raise RuntimeError(
            "Failed to patch pygfx line shader: expected position-load snippet not found"
        )
    code = code.replace(old_positions, new_positions)

    old_pick = """    varyings.pick_idx = u32(node_index);
    varyings.pick_zigzag = f32(node_index_is_even);"""
    new_pick = """    let pick_mapped_i = remap_multiline_index(node_index);
    varyings.pick_idx = u32(pick_mapped_i);
    varyings.pick_zigzag = f32(pick_mapped_i % 2 == 0);"""
    if old_pick not in code:
        raise RuntimeError(
            "Failed to patch pygfx line shader: expected picking snippet not found"
        )
    code = code.replace(old_pick, new_pick)

    replacements = [
        (
            "load_s_colors(node_index)",
            "load_s_colors(remap_multiline_index(node_index))",
            "colors(node_index)",
        ),
        (
            "load_s_texcoords(node_index)",
            "load_s_texcoords(remap_multiline_index(node_index))",
            "texcoords(node_index)",
        ),
        (
            "load_s_colors(select(node_index_prev, node_index_next, vertex_num >= 4))",
            "load_s_colors(remap_multiline_index(select(node_index_prev, node_index_next, vertex_num >= 4)))",
            "colors(select(prev,next))",
        ),
        (
            "load_s_texcoords(select(node_index_prev, node_index_next, vertex_num >= 4))",
            "load_s_texcoords(remap_multiline_index(select(node_index_prev, node_index_next, vertex_num >= 4)))",
            "texcoords(select(prev,next))",
        ),
    ]
    for old, new, label in replacements:
        if old not in code:
            raise RuntimeError(
                f"Failed to patch pygfx line shader: expected snippet not found for {label}"
            )
        code = code.replace(old, new)

    return code


_SCROLL_LINE_WGSL = _build_scroll_line_wgsl()


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
