import pytest
import os
import yaml

from sqlalchemy.orm import sessionmaker

from logtrucks.main import DetectLogic, GDriveDetectLogic
from logtrucks.schema import Detections, engine
from logtrucks.utils import get_uuid_from_filename


settings = yaml.safe_load(open("tests/test_settings.yaml"))


@pytest.fixture
def setup_db_session():
    session = sessionmaker(bind=engine)
    session = session()
    return session


@pytest.fixture
def teardown():
    yield
    files = os.listdir(settings["results_dir"])
    for file in files:
        os.remove(settings["results_dir"] + "/" + file)
    files = os.listdir(settings["download_directory"])
    for file in files:
        os.remove(settings["download_directory"] + "/" + file)


def test_reduce_detected_frames(teardown):
    detector = DetectLogic(settings=settings)
    detector.start_detections("tests/assets/clip2.mp4")
    files = os.listdir(settings["results_dir"])
    assert len(files) == 3


def test_if_invalid_iwpod_outputs_kept(teardown):
    detector = DetectLogic(settings=settings)
    detector.start_detections("tests/assets/iwpod_wrong_detection.jpg")
    files = os.listdir(settings["results_dir"])
    assert len(files) == 1


def test_if_valid_lps_saved(teardown):
    detector = DetectLogic(settings=settings)
    detector.start_detections("tests/assets/127.jpeg")
    files = os.listdir(settings["results_dir"])
    assert len(files) == 2


def test_write_all_results_to_db(teardown, setup_db_session):
    detector = DetectLogic(settings=settings)
    detector.start_detections("tests/assets/127.jpeg")
    session = setup_db_session
    files = os.listdir(detector.settings["results_dir"])
    for file in files:
        if file.endswith(".lpr.jpg"):
            lp = file.split("_")[1]
            result = session.query(Detections).filter(
                Detections.license_plate_prediction == lp).exists()
            assert session.query(result).scalar() is True


def test_process_new_items(teardown, setup_db_session):
    session = setup_db_session
    session.query(Detections).delete()
    session.commit()
    detector = GDriveDetectLogic(settings=settings)
    detector.process_new_items()
    uuids = set([uuid for filename in os.listdir(settings["results_dir"])
                 if (uuid := get_uuid_from_filename(filename)) is not None])
    assert len(uuids) == 1
    result = session.query(Detections).filter(
        Detections.id == list(uuids)[0])
    assert result.first().gdrive_file_id is not None
