import numpy as np
from numpy import testing as npt

from fastplotlib.graphics.features import MultiLinePositions


def test_multiline_positions_y_only():
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    positions = MultiLinePositions(data)

    assert positions.n_lines == 4
    assert positions.n_points == 5
    assert positions.value.shape == (4, 5, 3)

    expected_x = np.broadcast_to(
        np.arange(5, dtype=np.float32)[None, :], (4, 5)
    )
    npt.assert_almost_equal(positions.value[:, :, 0], expected_x)
    npt.assert_almost_equal(positions.value[:, :, 1], data)
    npt.assert_almost_equal(positions.value[:, :, 2], 0.0)

    packed = positions.flat_value.reshape(4, 6, 3)
    assert np.isnan(packed[:, -1, :]).all()
    assert np.shares_memory(positions.value, positions.flat_value)


def test_multiline_positions_xy_single_line():
    xs = np.linspace(0, 1, 6, dtype=np.float32)
    ys = np.sin(xs).astype(np.float32)
    data = np.column_stack([xs, ys])

    positions = MultiLinePositions(data)

    assert positions.n_lines == 1
    assert positions.n_points == 6
    assert positions.value.shape == (1, 6, 3)

    npt.assert_almost_equal(positions.value[0, :, 0], xs)
    npt.assert_almost_equal(positions.value[0, :, 1], ys)
    npt.assert_almost_equal(positions.value[0, :, 2], 0.0)


def test_multiline_positions_xyz_multi_line():
    data = np.random.rand(3, 4, 3).astype(np.float32)
    positions = MultiLinePositions(data)

    assert positions.n_lines == 3
    assert positions.n_points == 4

    npt.assert_almost_equal(positions.value, data)

    packed = positions.flat_value.reshape(3, 5, 3)
    assert np.isnan(packed[:, -1, :]).all()
