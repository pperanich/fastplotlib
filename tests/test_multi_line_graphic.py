import numpy as np
from numpy import testing as npt
import pytest
import pygfx
import pylinalg as la

import fastplotlib as fpl
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

    assert isinstance(graphic.data, MultiLinePositions)
    assert graphic.world_object.geometry.positions is graphic.data.buffer
    assert graphic.data.value.shape == (graphic.n_lines, graphic.n_points, 3)
    npt.assert_almost_equal(graphic.data.value[:, :, :2], data)
    npt.assert_almost_equal(graphic.data.value[:, :, 2], 0.0)

    new_y = -data[:, :, 1]
    graphic.data[:, :, 1] = new_y
    npt.assert_almost_equal(graphic.data.value[:, :, 1], new_y)

    packed = graphic.data.flat_value.reshape(
        graphic.n_lines, graphic.n_points + 1, 3
    )
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
    expected = np.repeat(
        line_colors[:, None, :], graphic.n_points + 1, axis=1
    ).reshape(-1, 4)
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

    assert la.mat_has_shear(graphic.world_object.local.matrix)
    npt.assert_almost_equal(graphic.world_object.local.matrix[1, 2], 1.0)
