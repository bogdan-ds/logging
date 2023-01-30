import os

import yaml

from logtrucks.utils import get_uuid_from_filename
from logtrucks.schema import Detections
from logtrucks.yolo.detect import YoloDetector
from tests.test_main import teardown, setup_db_session

settings = yaml.safe_load(open("tests/test_settings.yaml"))


def test_detect_logtruck(teardown):
    detector = YoloDetector(source="tests/assets/127.jpeg",
                            settings=settings)
    result = detector.detect(conf_thres=0.45)
    assert '1 logs' in result


def test_detect_no_logtruck(teardown):
    detector = YoloDetector(source="tests/assets/notruck.jpg",
                            settings=settings)
    result = detector.detect(conf_thres=0.45)
    assert '' == result


def test_frame_detect_logic(teardown):
    detector = YoloDetector(source="tests/assets/clip2.mp4",
                            settings=settings)
    detector.detect(conf_thres=0.45)
    uuids = set([get_uuid_from_filename(filename) for filename in
                 os.listdir(settings["results_dir"])])
    assert len(uuids) == 1


def test_write_yolo_detection_to_db(teardown, setup_db_session):
    detector = YoloDetector(source="tests/assets/127.jpeg",
                            settings=settings)
    detector.detect(conf_thres=0.45)
    session = setup_db_session
    uuids = set([get_uuid_from_filename(filename) for filename in
                 os.listdir(settings["results_dir"])])
    assert len(uuids) == 1
    result = session.query(Detections).filter(
        Detections.id == list(uuids)[0]).exists()
    assert session.query(result).scalar() is True


def test_write_to_db_setting(teardown, setup_db_session):
    settings["write_to_db"] = False
    detector = YoloDetector(source="tests/assets/127.jpeg",
                            settings=settings)
    detector.detect(conf_thres=0.45)
    session = setup_db_session
    uuids = set([get_uuid_from_filename(filename) for filename in
                 os.listdir(settings["results_dir"])])
    assert len(uuids) == 1
    result = session.query(Detections).filter(
        Detections.id == list(uuids)[0]).exists()
    assert session.query(result).scalar() is False


def test_frame_to_time(teardown, setup_db_session):
    settings["write_to_db"] = True
    detector = YoloDetector(source="tests/assets/clip2.mp4",
                            settings=settings)
    detector.detect(conf_thres=0.45)
    session = setup_db_session
    uuids = set([get_uuid_from_filename(filename) for filename in
                 os.listdir(settings["results_dir"])])
    assert len(uuids) == 1
    result = session.query(Detections).filter(
        Detections.id == list(uuids)[0])
    assert result.first().time_first_detected is not None
