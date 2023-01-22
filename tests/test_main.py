import pytest
import os

from sqlalchemy.orm import sessionmaker

from logtrucks.main import DetectLogic
from logtrucks.schema import Detections, engine


save_dir = "tests/assets/save_dir"


@pytest.fixture
def setup_db_session():
    session = sessionmaker(bind=engine)
    session = session()
    return session


@pytest.fixture
def teardown():
    yield
    files = os.listdir(save_dir)
    for file in files:
        os.remove(save_dir + "/" + file)


def test_reduce_detected_frames(teardown):
    detector = DetectLogic(settings="tests/test_settings.yaml")
    detector.start_detections("tests/assets/clip2.mp4")
    files = os.listdir(save_dir)
    assert len(files) == 3


def test_if_invalid_iwpod_outputs_kept(teardown):
    detector = DetectLogic(settings="tests/test_settings.yaml")
    detector.start_detections("tests/assets/iwpod_wrong_detection.jpg")
    files = os.listdir(save_dir)
    assert len(files) == 1


def test_if_valid_lps_saved(teardown):
    detector = DetectLogic(settings="tests/test_settings.yaml")
    detector.start_detections("tests/assets/127.jpeg")
    files = os.listdir(save_dir)
    assert len(files) == 2


def test_write_all_results_to_db(teardown, setup_db_session):
    detector = DetectLogic(settings="tests/test_settings.yaml")
    detector.start_detections("tests/assets/127.jpeg")
    session = setup_db_session
    files = os.listdir(detector.settings["results_dir"])
    for file in files:
        if file.endswith(".lpr.jpg"):
            lp = file.split("_")[1]
            result = session.query(Detections).filter(
                Detections.license_plate_prediction == lp).exists()
            assert session.query(result).scalar() is True
