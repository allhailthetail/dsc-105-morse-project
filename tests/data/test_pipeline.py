import pytest

tf = pytest.importorskip("tensorflow", reason="requires requirements-ml.txt (tensorflow) to be installed")

from morse.data.pipeline import make_dataset
from morse.features.melspec import N_MELS


def test_dataset_yields_consistent_shapes():
    ds = make_dataset(seed=0)
    batch = ds.padded_batch(4).take(1)
    for features, labels in batch:
        assert features.shape[0] == 4
        assert features.shape[2] == N_MELS
        assert labels.shape[0] == 4
        assert features.dtype == tf.float32
        assert labels.dtype == tf.int32
