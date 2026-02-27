import numpy as np
from numpy import testing as npt
import pytest
import pygfx
import pylinalg as la

import fastplotlib as fpl
from fastplotlib.graphics._multi_line_scroll import MultiLineScrollMaterial, _patch_line_shader
from fastplotlib.graphics.features import MultiLinePositions, VertexColors


def make_data(n_lines=3, n_points=6):
    xs = np.linspace(0, 2 * np.pi, n_points, dtype=np.float32)
    offsets = np.linspace(0, 2, n_lines, dtype=np.float32)
    data = np.zeros((n_lines, n_points, 2), dtype=np.float32)
    data[:, :, 0] = xs
    data[:, :, 1] = np.sin(xs)[None, :] + offsets[:, None]
    return data


def test_multi_line_graphic_buffer_and_update():
    data = make_data()
    fig = fpl.Figure()

    graphic = fig[0, 0].add_multi_line(data)
    line = graphic.world_object.children[0]

    assert isinstance(graphic.data, MultiLinePositions)
    assert isinstance(graphic.world_object, pygfx.Group)
    assert isinstance(line, pygfx.Line)
    assert line.geometry.positions is graphic.data.buffer
    assert graphic.data.value.shape == (graphic.n_lines, graphic.n_points, 3)
    npt.assert_almost_equal(graphic.data.value[:, :, :2], data)
    npt.assert_almost_equal(graphic.data.value[:, :, 2], 0.0)

    new_y = -data[:, :, 1]
    graphic.data[:, :, 1] = new_y
    npt.assert_almost_equal(graphic.data.value[:, :, 1], new_y)

    packed = graphic.data.flat_value.reshape(graphic.n_lines, graphic.n_points + 1, 3)
    assert np.isnan(packed[:, -1, :]).all()


def test_multi_line_graphic_colors_per_line():
    data = make_data(n_lines=3, n_points=4)
    colors = ["r", "g", "b"]

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, colors=colors)

    assert isinstance(graphic.colors, VertexColors)
    assert graphic.colors.value.shape == (
        graphic.n_lines * (graphic.n_points + 1),
        4,
    )

    line_colors = np.array([pygfx.Color(c) for c in colors], dtype=np.float32)
    expected = np.repeat(line_colors[:, None, :], graphic.n_points + 1, axis=1).reshape(
        -1, 4
    )
    npt.assert_almost_equal(graphic.colors.value, expected)


def test_multi_line_incompatible_cmap_uniform_color():
    data = make_data()
    fig = fpl.Figure()

    with pytest.raises(TypeError):
        fig[0, 0].add_multi_line(data, uniform_color=True, cmap="jet")


def test_multi_line_z_offset_shear():
    data = make_data()
    data = np.pad(data, ((0, 0), (0, 0), (0, 1)), mode="constant")
    data[:, :, 2] = np.linspace(0, 1, data.shape[0], dtype=np.float32)[:, None]

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, z_offset_scale=1.0)
    line = graphic.world_object.children[0]

    assert la.mat_has_shear(line.local.matrix)
    npt.assert_almost_equal(line.local.matrix[1, 2], 1.0)


def test_multi_line_z_offset_does_not_block_parent_offset():
    data = make_data()
    data = np.pad(data, ((0, 0), (0, 0), (0, 1)), mode="constant")
    data[:, :, 2] = np.linspace(0, 1, data.shape[0], dtype=np.float32)[:, None]

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, z_offset_scale=1.0)

    graphic.offset = (1.0, 2.0, 3.0)

    npt.assert_almost_equal(graphic.offset, (1.0, 2.0, 3.0))
    npt.assert_almost_equal(graphic.world_object.world.position, (1.0, 2.0, 3.0))


def test_multi_line_group_material_features():
    data = make_data()
    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, uniform_color=True, colors="w")
    line = graphic.world_object.children[0]

    graphic.alpha = 0.4
    graphic.thickness = 3.5
    graphic.size_space = "world"
    graphic.colors = "r"

    npt.assert_almost_equal(line.material.opacity, 0.4)
    npt.assert_almost_equal(line.material.thickness, 3.5)
    assert line.material.thickness_space == "world"
    npt.assert_almost_equal(
        np.asarray(line.material.color), np.asarray(pygfx.Color("r"))
    )


def test_multi_line_scroll_material_and_head():
    data = make_data(n_lines=2, n_points=8)
    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True, scroll_head=3, scroll_n_valid=5)
    line = graphic.world_object.children[0]

    assert isinstance(line.material, MultiLineScrollMaterial)
    assert graphic.scroll_enabled is True
    assert graphic.scroll_head == 3
    assert graphic.scroll_n_valid == 5

    graphic.set_scroll_head(2, n_valid=7)
    assert graphic.scroll_head == 2
    assert graphic.scroll_n_valid == 7


def test_multi_line_append_y_ring_updates():
    n_lines = 2
    n_points = 6
    data = np.zeros((n_lines, n_points), dtype=np.float32)

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True, scroll_n_valid=0)

    batch_a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )
    graphic.append_y(batch_a)
    assert graphic.scroll_head == 0
    assert graphic.scroll_n_valid == 4
    npt.assert_allclose(graphic.data.value[:, :4, 1], batch_a)

    batch_b = np.array(
        [
            [5.0, 6.0, 7.0],
            [50.0, 60.0, 70.0],
        ],
        dtype=np.float32,
    )
    graphic.append_y(batch_b)
    assert graphic.scroll_head == 1
    assert graphic.scroll_n_valid == n_points

    expected_physical = np.array(
        [
            [7.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [70.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ],
        dtype=np.float32,
    )
    npt.assert_allclose(graphic.data.value[:, :, 1], expected_physical)


def test_multi_line_scroll_supports_non_shared_x():
    data = np.zeros((2, 4, 2), dtype=np.float32)
    data[0, :, 0] = np.array([0, 1, 2, 3], dtype=np.float32)
    data[1, :, 0] = np.array([0, 2, 3, 5], dtype=np.float32)
    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True)
    line = graphic.world_object.children[0]
    assert isinstance(line.material, MultiLineScrollMaterial)
    assert line.material.scroll_use_shared_x is False


def test_multi_line_scroll_shader_patches_applied():
    from pygfx.renderers.wgpu import load_wgsl

    base_code = load_wgsl("line.wgsl")
    patched = _patch_line_shader(base_code)
    assert "remap_multiline_index" in patched
    assert "(col + u_material.scroll_head + n_valid) % n_points" in patched
    assert "pick_mapped_i = remap_multiline_index(node_index)" in patched
    assert "load_s_scroll_y" in patched
    assert "load_s_scroll_x" in patched


def test_multi_line_append_y_uses_update_range_not_full(monkeypatch):
    data = np.zeros((2, 8), dtype=np.float32)
    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True, scroll_n_valid=0)

    calls = {"range": 0, "full": 0}
    original_update_range = graphic.data.y_buffer.update_range
    original_update_full = graphic.data.y_buffer.update_full

    def _update_range(*, offset, size):
        calls["range"] += 1
        return original_update_range(offset=offset, size=size)

    def _update_full():
        calls["full"] += 1
        return original_update_full()

    monkeypatch.setattr(graphic.data.y_buffer, "update_range", _update_range)
    monkeypatch.setattr(graphic.data.y_buffer, "update_full", _update_full)

    y = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    graphic.append_y(y)

    assert calls["full"] == 0
    assert calls["range"] > 0


def test_multi_line_append_y_startup_respects_nonzero_head():
    n_lines = 2
    n_points = 6
    data = np.zeros((n_lines, n_points), dtype=np.float32)

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True, scroll_head=5, scroll_n_valid=0)

    y = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
    graphic.append_y(y)

    assert graphic.scroll_head == 5
    assert graphic.scroll_n_valid == 2
    npt.assert_allclose(graphic.data.value[:, 5, 1], np.array([1.0, 10.0], dtype=np.float32))
    npt.assert_allclose(graphic.data.value[:, 0, 1], np.array([2.0, 20.0], dtype=np.float32))


def test_multi_line_scroll_data_set_updates_scroll_x_buffer():
    n_lines = 2
    n_points = 5
    data = np.zeros((n_lines, n_points), dtype=np.float32)

    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=True)
    line = graphic.world_object.children[0]

    new_x = np.linspace(-1, 1, n_points, dtype=np.float32)
    new_y = np.arange(n_lines * n_points, dtype=np.float32).reshape(n_lines, n_points)
    new_data = np.zeros((n_lines, n_points, 2), dtype=np.float32)
    new_data[:, :, 0] = new_x
    new_data[:, :, 1] = new_y

    graphic.data = new_data
    npt.assert_allclose(line.material.scroll_x_buffer.data, new_x)


def test_multi_line_data_set_non_shared_x_disables_shared_x_buffering():
    n_lines = 2
    n_points = 5
    data = np.zeros((n_lines, n_points), dtype=np.float32)
    fig = fpl.Figure()
    graphic = fig[0, 0].add_multi_line(data, scroll=False)
    line = graphic.world_object.children[0]

    new_data = np.zeros((n_lines, n_points, 2), dtype=np.float32)
    new_data[0, :, 0] = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    new_data[1, :, 0] = np.array([0, 2, 4, 6, 8], dtype=np.float32)
    new_data[:, :, 1] = np.array(
        [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float32
    )

    graphic.data = new_data
    assert isinstance(line.material, MultiLineScrollMaterial)
    assert line.material.scroll_use_shared_x is False
