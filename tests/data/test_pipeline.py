import pytest

tf = pytest.importorskip("tensorflow", reason="requires requirements-ml.txt (tensorflow) to be installed")

from morse.data.pipeline import make_dataset, prepare_for_training
from morse.features.melspec import N_MELS


def test_dataset_yields_consistent_shapes():
    ds = make_dataset(seed=0)
    batch = prepare_for_training(ds, batch_size=4).take(1)
    for features, labels, input_length, label_length in batch:
        assert features.shape[0] == 4
        assert features.shape[2] == N_MELS
        assert labels.shape[0] == 4
        assert input_length.shape == (4,)
        assert label_length.shape == (4,)
        assert features.dtype == tf.float32
        assert labels.dtype == tf.int32
        assert input_length.dtype == tf.int32
        assert label_length.dtype == tf.int32


def test_lengths_match_unpadded_content():
    ds = make_dataset(seed=1)
    for features, label_ids, input_length, label_length in ds.take(5):
        assert int(input_length) == features.shape[0]
        assert int(label_length) == label_ids.shape[0]


def test_curriculum_ranges_are_forwarded():
    # A narrow, fixed WPM range should make every drawn example's message
    # render to a duration consistent with that WPM (loosely -- humanization
    # jitter/QSB doesn't change the *label* length distribution here).
    ds = make_dataset(seed=2, wpm_range=(20.0, 20.0), length_range=(15, 15))
    for _, label_ids, _, label_length in ds.take(3):
        assert label_ids.shape[0] == 15
        assert int(label_length) == 15


def test_padded_batch_pads_shorter_examples_to_batch_max():
    ds = make_dataset(seed=3)
    batch = prepare_for_training(ds, batch_size=8).take(1)
    for features, labels, input_length, label_length in batch:
        # Every row is padded up to the batch's max length, but the true
        # lengths (captured before padding) still reflect each example's own size.
        assert int(tf.reduce_max(input_length)) <= features.shape[1]
        assert int(tf.reduce_max(label_length)) <= labels.shape[1]
