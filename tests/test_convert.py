import json
import os
import zipfile

from click.testing import CliRunner

from poseval import eval_helpers
from poseval.convert import cli, convert_videos
from poseval.evaluateAP import evaluateAP
from poseval.posetrack18_id2fname import posetrack18_fname2id, posetrack18_id2fname

from conftest import N_FRAMES, N_JOINTS, N_PEOPLE, SEQ_NAMES, make_sequence


def test_fname_id_roundtrip():
    image_id = posetrack18_fname2id(SEQ_NAMES[0], 3)
    assert posetrack18_id2fname(image_id) == (SEQ_NAMES[0], 3)


def test_convert_old_to_new():
    converted = convert_videos(make_sequence(SEQ_NAMES[0], with_scores=True))
    assert len(converted) == 1
    new = converted[0]
    assert len(new["images"]) == N_FRAMES
    assert len(new["annotations"]) == N_FRAMES * N_PEOPLE
    assert len(new["categories"][0]["keypoints"]) == 17
    for annotation in new["annotations"]:
        assert len(annotation["keypoints"]) == 17 * 3
        assert len(annotation["scores"]) == 17


def test_convert_roundtrip():
    old = make_sequence(SEQ_NAMES[0], with_scores=True)
    roundtrip = convert_videos(convert_videos(old)[0])[0]
    assert len(roundtrip["annolist"]) == N_FRAMES
    for frame_old, frame_new in zip(old["annolist"], roundtrip["annolist"]):
        assert frame_old["image"] == frame_new["image"]
        assert frame_old["imgnum"] == frame_new["imgnum"]
        assert len(frame_new["annorect"]) == N_PEOPLE
        for rect_old, rect_new in zip(frame_old["annorect"], frame_new["annorect"]):
            for key in ["track_id", "x1", "y1", "x2", "y2"]:
                assert rect_old[key] == rect_new[key]
            points_old = {p["id"][0]: p for p in rect_old["annopoints"][0]["point"]}
            points_new = {p["id"][0]: p for p in rect_new["annopoints"][0]["point"]}
            assert sorted(points_new.keys()) == list(range(N_JOINTS))
            for joint_idx, point_old in points_old.items():
                for key in ["x", "y", "score"]:
                    assert point_old[key][0] == points_new[joint_idx][key][0]


def test_cli_directory_both_directions(gt_dir, tmp_path):
    runner = CliRunner()
    new_dir = str(tmp_path / "new")
    result = runner.invoke(cli, [gt_dir, "--out_fp", new_dir])
    assert result.exit_code == 0, result.output
    assert sorted(os.listdir(new_dir)) == [name + ".json" for name in SEQ_NAMES]
    with open(os.path.join(new_dir, SEQ_NAMES[0] + ".json")) as inf:
        assert "images" in json.load(inf)

    old_dir = str(tmp_path / "old")
    result = runner.invoke(cli, [new_dir, "--out_fp", old_dir])
    assert result.exit_code == 0, result.output
    with open(os.path.join(old_dir, SEQ_NAMES[0] + ".json")) as inf:
        assert len(json.load(inf)["annolist"]) == N_FRAMES


def test_cli_zip(gt_dir, tmp_path):
    zip_fp = str(tmp_path / "gt.zip")
    with zipfile.ZipFile(zip_fp, "w") as zip_file:
        for fname in os.listdir(gt_dir):
            zip_file.write(os.path.join(gt_dir, fname), fname)
    out_dir = str(tmp_path / "converted")
    result = CliRunner().invoke(cli, [zip_fp, "--out_fp", out_dir])
    assert result.exit_code == 0, result.output
    assert sorted(os.listdir(out_dir)) == [name + ".json" for name in SEQ_NAMES]


def test_evaluate_posetrack18_format(gt_dir, pred_dir, tmp_path):
    """Files in PoseTrack18 format are converted on the fly."""
    runner = CliRunner()
    gt_new = str(tmp_path / "gt_new")
    pred_new = str(tmp_path / "pred_new")
    assert runner.invoke(cli, [gt_dir, "--out_fp", gt_new]).exit_code == 0
    assert runner.invoke(cli, [pred_dir, "--out_fp", pred_new]).exit_code == 0

    gt_frames, pr_frames = eval_helpers.load_data_dir(['', gt_new + "/", pred_new + "/"])
    assert len(gt_frames) == len(SEQ_NAMES) * N_FRAMES
    ap, _, _ = evaluateAP(gt_frames, pr_frames, str(tmp_path), False, False)
    assert (ap == 100.0).all()
