"""Manifest, duplicate, and frame-sampling tests (PREPROCESSING.md 19.2, 19.3).

The frame tests need the restricted dataset and skip themselves without it, so
the label suite still runs on a machine that has no access to the video.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.labels import Code
from src.data.preprocess import detect_duplicates, probe_clip, resolve_clip_path
from src.features.extract import sample_window

SAMPLE_CLIP = config.DATA_ROOT / "mvfouls" / "Train" / "action_0" / "clip_0.mp4"
needs_dataset = pytest.mark.skipif(
    not SAMPLE_CLIP.exists(), reason="restricted dataset not present"
)


# ------------------------------------------------------- path resolution

def test_annotated_url_is_preferred_over_a_rebuilt_path(tmp_path):
    """Clip indices happen to be contiguous here, but that stays an observation."""
    got = resolve_clip_path(tmp_path, "7", 0, "Dataset/Train/action_7/clip_3")
    assert got == tmp_path / "action_7" / "clip_3.mp4"


def test_path_falls_back_when_the_annotation_has_no_url(tmp_path):
    assert resolve_clip_path(tmp_path, "7", 2, None) == tmp_path / "action_7" / "clip_2.mp4"


def test_backslash_urls_resolve_the_same_as_forward_slashes(tmp_path):
    a = resolve_clip_path(tmp_path, "7", 0, "Dataset\\Train\\action_7\\clip_1")
    b = resolve_clip_path(tmp_path, "7", 0, "Dataset/Train/action_7/clip_1")
    assert a == b


# ------------------------------------------------------------- duplicates

def _clip_row(split, action, index, path, size):
    return {
        "file_exists": True, "file_size_bytes": size, "split": split,
        "action_key": f"{split}:{action}", "clip_index": index,
        "resolved_path": str(path),
    }


def test_identical_bytes_across_splits_is_a_hard_failure(tmp_path):
    """A train clip reappearing in test invalidates every number we report."""
    payload = b"identical video bytes"
    train, test = tmp_path / "a.mp4", tmp_path / "b.mp4"
    train.write_bytes(payload)
    test.write_bytes(payload)

    report, leak = detect_duplicates(pd.DataFrame([
        _clip_row("train", "1", 0, train, len(payload)),
        _clip_row("test", "2", 0, test, len(payload)),
    ]))
    assert leak is True
    assert set(report["reason_code"]) == {Code.CROSS_SPLIT_DUPLICATE}


def test_duplicate_within_one_action_is_flagged_but_not_a_leak(tmp_path):
    payload = b"same replay twice"
    first, second = tmp_path / "c0.mp4", tmp_path / "c1.mp4"
    first.write_bytes(payload)
    second.write_bytes(payload)

    report, leak = detect_duplicates(pd.DataFrame([
        _clip_row("train", "1", 0, first, len(payload)),
        _clip_row("train", "1", 1, second, len(payload)),
    ]))
    assert leak is False
    assert set(report["reason_code"]) == {Code.DUPLICATE_WITHIN_ACTION}


def test_same_size_different_bytes_is_not_a_duplicate(tmp_path):
    """Size collision is the cheap filter, not the verdict."""
    first, second = tmp_path / "x.mp4", tmp_path / "y.mp4"
    first.write_bytes(b"aaaaaaaa")
    second.write_bytes(b"bbbbbbbb")

    report, leak = detect_duplicates(pd.DataFrame([
        _clip_row("train", "1", 0, first, 8),
        _clip_row("valid", "2", 0, second, 8),
    ]))
    assert leak is False
    assert report.empty


def test_no_shared_sizes_skips_hashing_entirely():
    report, leak = detect_duplicates(pd.DataFrame([
        _clip_row("train", "1", 0, "a.mp4", 100),
        _clip_row("train", "2", 0, "b.mp4", 200),
    ]))
    assert report.empty and leak is False


# -------------------------------------------------------- media validation

def test_a_missing_file_is_reported_not_raised(tmp_path):
    info = probe_clip(tmp_path / "nope.mp4", decode=False)
    assert info["file_exists"] is False
    assert Code.MISSING_CLIP in info["codes"]
    assert info["decodable"] is False


def test_a_non_video_file_is_rejected(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert probe_clip(junk, decode=False)["decodable"] is False


# --------------------------------------------------------- frame sampling

@needs_dataset
def test_sampler_returns_the_documented_shape_and_dtype():
    frames = sample_window(SAMPLE_CLIP)
    assert frames.shape[0] == config.NUM_FRAMES
    assert frames.shape[-1] == 3
    assert frames.dtype == np.uint8


@needs_dataset
def test_sampling_is_deterministic():
    """Validation and test preprocessing must repeat exactly (spec section 15)."""
    assert np.array_equal(sample_window(SAMPLE_CLIP), sample_window(SAMPLE_CLIP))


@needs_dataset
def test_frames_are_rgb_not_bgr():
    """cv2 decodes BGR; a missed conversion would silently swap the channels."""
    import cv2

    cap = cv2.VideoCapture(str(SAMPLE_CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.START_FRAME)
    ok, raw_bgr = cap.read()
    cap.release()
    assert ok

    first = sample_window(SAMPLE_CLIP)[0]
    assert np.array_equal(first, cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))


@needs_dataset
def test_a_wider_window_reaches_further_into_the_clip():
    """Guards the 43-107 change: a wider window must not return the same frames
    as the narrow one, which is what near-duplicate sampling looked like."""
    narrow = sample_window(SAMPLE_CLIP, start=63, end=87)
    wide = sample_window(SAMPLE_CLIP, start=43, end=107)
    assert not np.array_equal(narrow, wide)


@needs_dataset
def test_window_past_the_end_of_a_clip_degrades_instead_of_crashing():
    frames = sample_window(SAMPLE_CLIP, start=10_000, end=10_016)
    assert frames.shape[0] == config.NUM_FRAMES
