import os

from logtrucks.main import DetectLogic


save_dir = "tests/assets/save_dir"


def test_lp_validation():
    valid = ["CA 3812 MA", "C 3812 HH"]
    invalid = ["AZ 3812 HH", "A 3300 M", "T 338 AM"]
    for lp in valid:
        assert DetectLogic.is_lp_valid(lp) is True
    for lp in invalid:
        assert DetectLogic.is_lp_valid(lp) is not True


def test_reduce_detected_frames():
    detector = DetectLogic(settings="tests/test_settings.yaml")
    detector.start_detections("tests/assets/clip2.mp4")
    files = os.listdir(save_dir)
    assert len(files) == 3
    for file in files:
        os.remove(save_dir + "/" + file)


def test_if_invalid_iwpod_outputs_are_kept():
    pass
