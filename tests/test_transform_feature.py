import numpy as np
from numpy import testing as npt
import pytest
import pylinalg as la

import fastplotlib as fpl


def test_transform_sets_matrix_and_blocks_components():
    data = np.random.rand(10).astype(np.float32)
    graphic = fpl.LineGraphic(data)

    shear = np.eye(4, dtype=np.float32)
    shear[1, 2] = 0.75

    graphic.transform = shear
    npt.assert_almost_equal(graphic.transform, shear)
    assert la.mat_has_shear(graphic.world_object.local.matrix)

    with pytest.raises(RuntimeError):
        graphic.offset = (1.0, 2.0, 3.0)
