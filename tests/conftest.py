import json

import pytest

SEQ_NAMES = ["008838_mpii_train", "012218_mpii_train"]
N_FRAMES = 4
N_PEOPLE = 2
N_JOINTS = 15


def make_sequence(seq_name, with_scores=False, offset=0):
    """Create a synthetic sequence in PoseTrack17 format."""
    annolist = []
    for frame_idx in range(N_FRAMES):
        rects = []
        for track_id in range(N_PEOPLE):
            base_x = 100 + 200 * track_id + 5 * frame_idx
            base_y = 100
            points = []
            for joint_idx in range(N_JOINTS):
                point = {
                    "id": [joint_idx],
                    "x": [base_x + 7 * joint_idx + offset],
                    "y": [base_y + 11 * joint_idx],
                }
                if with_scores:
                    point["score"] = [0.9]
                points.append(point)
            rects.append(
                {
                    "track_id": [track_id],
                    "x1": [base_x],
                    "y1": [base_y],
                    "x2": [base_x + 40],
                    "y2": [base_y + 40],
                    "annopoints": [{"point": points}],
                }
            )
        annolist.append(
            {
                "image": [{"name": "images/%s/%06d.jpg" % (seq_name, frame_idx)}],
                "imgnum": [frame_idx + 1],
                "annorect": rects,
            }
        )
    return {"annolist": annolist}


def write_sequences(directory, **kwargs):
    directory.mkdir()
    for seq_name in SEQ_NAMES:
        with open(str(directory / (seq_name + ".json")), "w") as outf:
            json.dump(make_sequence(seq_name, **kwargs), outf)
    # load_data_dir concatenates directory and file name
    return str(directory) + "/"


@pytest.fixture
def gt_dir(tmp_path):
    return write_sequences(tmp_path / "gt")


@pytest.fixture
def pred_dir(tmp_path):
    """Predictions that are identical to the ground truth."""
    return write_sequences(tmp_path / "pred", with_scores=True)
