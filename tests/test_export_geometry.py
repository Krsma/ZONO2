import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pzr.learning.monitor import Zonotope

TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools.learning.export_geometry import *

"""
class GeometrySample:
    trace_id: str
    sample_id: str
    step: int
    teacher_action: str
    features: np.ndarray
    dynamic: Zonotope
    total: Zonotope
"""

def _sample(i: int, d_gen: int, t_gen:int) -> GeometrySample:
    d = 2
    return GeometrySample(trace_id=f"trace-{i}", sample_id=f"sample-{i}", step=i,
                          teacher_action="girard",
                          features=np.arange(0, 15, 1, dtype=float) + 10 * i,
                          dynamic=Zonotope(
                                    center=np.arange(0, d, 1, dtype=float) + 100 * i,
                                    generators=np.arange(d*d_gen, dtype=float).reshape(d, d_gen) + 1000 * i
                                ),
                          total=Zonotope(
                                    center=np.arange(0, d, 1, dtype=float) - 100 * i,
                                    generators=np.arange(d*t_gen, dtype=float).reshape(d, t_gen) - 1000 * i
                                )
                          )


def _dataset() -> GeometrySampleDataset:
    return GeometrySampleDataset((
        _sample(i=0, d_gen=1, t_gen=3),
        _sample(i=1, d_gen=4, t_gen=0),
        _sample(i=2, d_gen=0, t_gen=2))
    )


def test_replaying_traces_save_geometry(tmp_path: Path):
    save_geometry(_dataset(), tmp_path, metadata={"complete": True})
    loaded, metadata, manifest = load_geometry(tmp_path)
    assert manifest["schema"] == GEOMETRY_SCHEMA
    assert manifest["num_samples"] == 3
    for original, restored in zip(_dataset().samples, loaded.samples):
        assert restored.sample_id == original.sample_id
        assert restored.step == original.step
        np.testing.assert_array_equal(restored.features, original.features)
        np.testing.assert_array_equal(restored.dynamic.center, original.dynamic.center)
        np.testing.assert_array_equal(restored.dynamic.generators, original.dynamic.generators)
        np.testing.assert_array_equal(restored.total.generators, original.total.generators)
