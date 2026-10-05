import json

import numpy as np
import pytest

from poseval import eval_helpers

from conftest import N_FRAMES, N_JOINTS, SEQ_NAMES, make_sequence


def test_get_head_size():
    assert eval_helpers.getHeadSize(0, 0, 3, 4) == pytest.approx(3.0)


def test_compute_rpc():
    scores = np.array([0.9, 0.1, 0.8, 0.5])
    labels = np.array([1, 0, 0, 1])
    precision, recall, idxs = eval_helpers.computeRPC(scores, labels, 2)
    assert idxs.tolist() == [0, 2, 3, 1]
    np.testing.assert_allclose(precision, [1.0, 0.5, 2.0 / 3.0, 0.5])
    np.testing.assert_allclose(recall, [0.5, 0.5, 1.0, 1.0])


def test_voc_ap():
    assert eval_helpers.VOCap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert eval_helpers.VOCap(np.array([0.5, 0.5]), np.array([1.0, 0.5])) == pytest.approx(0.5)


def test_load_data_dir(gt_dir, pred_dir):
    gt_frames, pr_frames = eval_helpers.load_data_dir(['', gt_dir, pred_dir])
    assert len(gt_frames) == len(pr_frames) == len(SEQ_NAMES) * N_FRAMES
    assert {frame["seq_name"] for frame in gt_frames} == set(SEQ_NAMES)
    # track ids are unique across sequences
    track_ids = {
        frame["seq_id"]: {rect["track_id"][0] for rect in frame["annorect"]}
        for frame in gt_frames
    }
    assert not track_ids[0] & track_ids[1]


def test_load_data_dir_non_unique_track_ids(gt_dir, pred_dir):
    filename = pred_dir + SEQ_NAMES[0] + ".json"
    with open(filename) as inf:
        data = json.load(inf)
    data["annolist"][0]["annorect"][1]["track_id"] = [0]
    with open(filename, "w") as outf:
        json.dump(data, outf)
    with pytest.raises(Exception, match="Non-unique tracklet IDs"):
        eval_helpers.load_data_dir(['', gt_dir, pred_dir])


def test_remove_ignored_points():
    gt_frames = make_sequence(SEQ_NAMES[0])["annolist"][:1]
    pr_frames = make_sequence(SEQ_NAMES[0], with_scores=True)["annolist"][:1]
    # region that covers the first joint of the first person only
    gt_frames[0]["ignore_regions"] = [
        {"point": [{"x": [x], "y": [y]} for x, y in [(90, 90), (105, 90), (105, 105), (90, 105)]]}
    ]
    gt_frames, pr_frames = eval_helpers.removeIgnoredPoints(gt_frames, pr_frames)
    for frames in [gt_frames, pr_frames]:
        points = frames[0]["annorect"][0]["annopoints"][0]["point"]
        assert [point["id"][0] for point in points] == list(range(1, N_JOINTS))
        assert len(frames[0]["annorect"][1]["annopoints"][0]["point"]) == N_JOINTS
