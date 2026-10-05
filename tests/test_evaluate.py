import json
import os

import numpy as np

from poseval import eval_helpers
from poseval.convert import convert_videos
from poseval.evaluateAP import evaluateAP
from poseval.evaluatePCKh import evaluatePCKh
from poseval.evaluateTracking import evaluateTracking

from conftest import N_JOINTS, SEQ_NAMES, make_sequence


def test_evaluate_ap_perfect_predictions(gt_dir, pred_dir, tmp_path):
    gt_frames, pr_frames = eval_helpers.load_data_dir(['', gt_dir, pred_dir])
    ap, pre, rec = evaluateAP(gt_frames, pr_frames, str(tmp_path), True, True)
    assert ap.shape == pre.shape == rec.shape == (N_JOINTS + 1, 1)
    np.testing.assert_allclose(ap, 100.0)
    np.testing.assert_allclose(pre, 100.0)
    np.testing.assert_allclose(rec, 100.0)

    for name in ["total"] + SEQ_NAMES:
        with open(os.path.join(str(tmp_path), name + "_AP_metrics.json")) as inf:
            metrics = json.load(inf)
        np.testing.assert_allclose(metrics["ap"], 100.0)


def test_evaluate_ap_wrong_predictions(tmp_path):
    gt_frames = make_sequence(SEQ_NAMES[0])["annolist"]
    pr_frames = make_sequence(SEQ_NAMES[0], with_scores=True, offset=1000)["annolist"]
    for frames in [gt_frames, pr_frames]:
        for frame in frames:
            frame["seq_id"] = 0
            frame["seq_name"] = SEQ_NAMES[0]
    ap, _, rec = evaluateAP(gt_frames, pr_frames, str(tmp_path), False, False)
    np.testing.assert_allclose(ap, 0.0)
    np.testing.assert_allclose(rec, 0.0)


def test_evaluate_tracking_perfect_predictions(gt_dir, pred_dir, tmp_path):
    gt_frames, pr_frames = eval_helpers.load_data_dir(['', gt_dir, pred_dir])
    metrics = evaluateTracking(gt_frames, pr_frames, str(tmp_path), True, True)
    for name in ["mota", "motp", "pre", "rec"]:
        assert metrics[name].shape == (1, N_JOINTS + 1)
        np.testing.assert_allclose(metrics[name], 100.0)

    for name in ["total"] + SEQ_NAMES:
        with open(os.path.join(str(tmp_path), name + "_MOT_metrics.json")) as inf:
            saved = json.load(inf)
        np.testing.assert_allclose(saved["mota"], 100.0)


def test_evaluate_tracking_id_switch(gt_dir, pred_dir, tmp_path):
    filename = pred_dir + SEQ_NAMES[0] + ".json"
    with open(filename) as inf:
        data = json.load(inf)
    # swap the track ids of the two people in the second frame
    for rect in data["annolist"][1]["annorect"]:
        rect["track_id"][0] = 1 - rect["track_id"][0]
    with open(filename, "w") as outf:
        json.dump(data, outf)

    gt_frames, pr_frames = eval_helpers.load_data_dir(['', gt_dir, pred_dir])
    metrics = evaluateTracking(gt_frames, pr_frames, str(tmp_path), False, False)
    assert (metrics["mota"] < 100.0).all()
    np.testing.assert_allclose(metrics["pre"], 100.0)
    np.testing.assert_allclose(metrics["rec"], 100.0)


def test_evaluate_pckh():
    gt_frames = make_sequence(SEQ_NAMES[0])["annolist"]
    pck = evaluatePCKh(gt_frames, make_sequence(SEQ_NAMES[0])["annolist"])
    assert pck.shape == (N_JOINTS + 1, 1)
    np.testing.assert_allclose(pck, 100.0)

    pck = evaluatePCKh(gt_frames, make_sequence(SEQ_NAMES[0], offset=1000)["annolist"])
    np.testing.assert_allclose(pck, 0.0)


def test_evaluate_pckh_numpy_coordinates():
    """Coordinates of in-memory converted data are numpy scalars."""
    gt = make_sequence(SEQ_NAMES[0])
    pr_frames = convert_videos(convert_videos(make_sequence(SEQ_NAMES[0]))[0])[0]["annolist"]
    x = pr_frames[0]["annorect"][0]["annopoints"][0]["point"][0]["x"][0]
    assert isinstance(x, np.integer)
    np.testing.assert_allclose(evaluatePCKh(gt["annolist"], pr_frames), 100.0)
