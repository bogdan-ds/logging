import os

import torch
import yaml

from logtrucks.yolo.detect import YoloDetector

settings = yaml.safe_load(open("tests/test_settings.yaml"))


def test_detect_logtruck():
    with torch.no_grad():
        detector = YoloDetector(source="tests/assets/127.jpeg",
                                weights=settings["yolo_weights"],
                                save_dir=settings["results_dir"])
        result = detector.detect(conf_thres=0.45)
        assert '1 logs' in result


def test_detect_no_logtruck():
    with torch.no_grad():
        detector = YoloDetector(source="tests/assets/notruck.jpg",
                                weights=settings["yolo_weights"],
                                save_dir=settings["results_dir"])
        result = detector.detect(conf_thres=0.45)
        assert '' == result


def test_frame_detect_logic():
    with torch.no_grad():
        detector = YoloDetector(source="tests/assets/clip2.mp4",
                                weights=settings["yolo_weights"],
                                save_dir=settings["results_dir"])
        detector.detect(conf_thres=0.45)
    uuids = set([filename.split("_")[0] for filename in
                 os.listdir(settings["results_dir"])])
    assert len(uuids) == 1
    for filename in os.listdir(settings["results_dir"]):
        os.remove(settings["results_dir"] + "/" + filename)
