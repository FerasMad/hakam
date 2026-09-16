"""Section 19.3: augmentation must be temporally consistent and eval-safe."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import transforms as tf


def clip(t: int = 16, h: int = 64, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def test_eval_splits_are_never_augmented():
    rng = np.random.default_rng(1)
    for split in ("valid", "test", "challenge"):
        assert tf.build_params(tf.MILD_V1, split, rng, (64, 96)) is None


def test_no_spec_means_no_augmentation():
    rng = np.random.default_rng(1)
    assert tf.build_params(None, "train", rng, (64, 96)) is None


def test_eval_output_is_identical_across_runs():
    frames = clip()
    a = tf.apply_clip(frames, tf.build_params(tf.MILD_V1, "test", np.random.default_rng(1), (64, 96)))
    b = tf.apply_clip(frames, tf.build_params(tf.MILD_V1, "test", np.random.default_rng(9), (64, 96)))
    assert np.array_equal(a, b)
    assert np.array_equal(a, frames)


def test_train_augmentation_actually_changes_pixels():
    frames = clip()
    params = tf.build_params(tf.MILD_V1, "train", np.random.default_rng(3), (64, 96))
    assert params is not None
    assert not np.array_equal(tf.apply_clip(frames, params), frames)


def test_shape_and_dtype_survive():
    frames = clip()
    params = tf.build_params(tf.MILD_V1, "train", np.random.default_rng(4), (64, 96))
    out = tf.apply_clip(frames, params)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8


def test_one_spatial_transform_for_the_whole_clip():
    """The point of section 16: per-frame crops invent camera motion.

    Feeding sixteen identical frames must give sixteen identical outputs. If
    the crop were resampled per frame they would drift apart.
    """
    single = clip(t=1)
    frames = np.repeat(single, 16, axis=0)
    params = tf.build_params(tf.MILD_V1, "train", np.random.default_rng(5), (64, 96))
    out = tf.apply_clip(frames, params)
    for i in range(1, len(out)):
        assert np.array_equal(out[0], out[i]), f"frame {i} differs - transform is per-frame"


def test_horizontal_flip_disabled_when_direction_matters():
    spec = tf.direction_safe(tf.MILD_V1)
    for seed in range(30):
        params = tf.build_params(spec, "train", np.random.default_rng(seed), (64, 96))
        assert params["hflip"] is False


def test_temporal_jitter_stays_inside_the_clip():
    rng = np.random.default_rng(7)
    for _ in range(200):
        start, end = tf.jittered_window(tf.MILD_V1, "train", rng, total_frames=126)
        assert 0 <= start < end <= 126
        assert end - start == 64


def test_temporal_jitter_is_bounded_by_the_spec():
    rng = np.random.default_rng(8)
    shifts = {tf.jittered_window(tf.MILD_V1, "train", rng, 126)[0] for _ in range(300)}
    assert min(shifts) >= 43 - tf.MILD_V1.temporal_jitter
    assert max(shifts) <= 43 + tf.MILD_V1.temporal_jitter


def test_eval_window_is_not_jittered():
    rng = np.random.default_rng(9)
    for split in ("valid", "test"):
        assert tf.jittered_window(tf.MILD_V1, split, rng, 126) == (43, 107)


def test_cache_identities_are_distinct():
    assert tf.cache_tag("train") == "train__baseline"
    assert tf.cache_tag("train", tf.MILD_V1) == "train__mild_aug_v1"
    assert tf.cache_tag("train") != tf.cache_tag("train", tf.MILD_V1)


def test_describe_names_the_recipe():
    assert "no augmentation" in tf.describe(None)
    assert "mild_aug_v1" in tf.describe(tf.MILD_V1)
    assert "hflip_p=0.0" in tf.describe(tf.direction_safe(tf.MILD_V1))
