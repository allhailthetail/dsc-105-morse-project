import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="requires requirements-ml.txt (tensorflow) to be installed")
keras = pytest.importorskip("keras")

from morse.labels.vocab import VOCAB_SIZE
from morse.models.crnn import build_crnn, ctc_loss, greedy_decode


def test_output_shape():
    model = build_crnn(n_mels=40)
    batch, time = 2, 50
    x = np.random.default_rng(0).normal(size=(batch, time, 40)).astype(np.float32)
    logits = model(x, training=False)
    assert logits.shape == (batch, time, VOCAB_SIZE)


def test_time_axis_is_preserved_by_conv_stack():
    # Central assumption documented in crnn.py: output time == input time,
    # since pooling only ever happens over the frequency axis.
    model = build_crnn(n_mels=40)
    for time in (17, 50, 123):
        x = np.zeros((1, time, 40), dtype=np.float32)
        logits = model(x, training=False)
        assert logits.shape[1] == time


def test_ctc_loss_runs_and_is_finite():
    model = build_crnn(n_mels=40)
    batch, time, label_len = 3, 40, 5
    rng = np.random.default_rng(1)
    x = rng.normal(size=(batch, time, 40)).astype(np.float32)
    labels = rng.integers(0, VOCAB_SIZE - 1, size=(batch, label_len)).astype(np.int32)  # never emit blank as a "true" label
    input_length = np.full((batch,), time, dtype=np.int32)
    label_length = np.full((batch,), label_len, dtype=np.int32)

    logits = model(x, training=False)
    loss = ctc_loss(labels, logits, label_length, input_length)
    loss_np = keras.ops.convert_to_numpy(loss)
    assert loss_np.shape == (batch,)
    assert np.all(np.isfinite(loss_np))


def test_ctc_loss_is_differentiable():
    model = build_crnn(n_mels=40, conv_filters=(8,), lstm_units=16, num_lstm_layers=1)
    batch, time, label_len = 2, 30, 4
    rng = np.random.default_rng(2)
    x = tf.constant(rng.normal(size=(batch, time, 40)).astype(np.float32))
    labels = tf.constant(rng.integers(0, VOCAB_SIZE - 1, size=(batch, label_len)).astype(np.int32))
    input_length = tf.constant(np.full((batch,), time, dtype=np.int32))
    label_length = tf.constant(np.full((batch,), label_len, dtype=np.int32))

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(ctc_loss(labels, logits, label_length, input_length))
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)


def test_greedy_decode_runs_and_returns_strings():
    model = build_crnn(n_mels=40)
    batch, time = 2, 60
    x = np.random.default_rng(3).normal(size=(batch, time, 40)).astype(np.float32)
    logits = model(x, training=False)
    input_length = np.full((batch,), time, dtype=np.int32)

    texts = greedy_decode(logits, input_length)
    assert len(texts) == batch
    assert all(isinstance(t, str) for t in texts)
