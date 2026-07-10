from experiments.fgpl_seed import score

ROWS = [{"pano_name": "u1", "room": "office_4"}, {"pano_name": "u2", "room": "office_6"}]
GT = {"u1": {"location": [1, 2, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]]},
      "u2": {"location": [8, 9, 1.4], "R_cw": [[1,0,0],[0,1,0],[0,0,1]]}}
CENTS = {"office_4": [1, 2, 1.4], "office_6": [8, 9, 1.4]}

def test_perfect_poses_zero_error_and_no_wrong_room():
    poses = {"u1": {"translation": [1, 2, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]},
             "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["translation"]["max"] < 1e-6
    assert s["wrong_room_rate"] == 0.0
    assert s["n_localized"] == 2

def test_swapped_pose_counts_as_wrong_room():
    poses = {"u1": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]},  # in office_6
             "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["wrong_room_rate"] == 0.5           # u1 landed in the wrong room

def test_none_pose_excluded_from_localized():
    poses = {"u1": None, "u2": {"translation": [8, 9, 1.4], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}}
    s = score.score_arm(poses, GT, ROWS, CENTS)
    assert s["n_localized"] == 1
