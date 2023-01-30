import os
import yaml

import pytest

from logtrucks.gdrive_client import GDriveClient


settings = yaml.safe_load(open("tests/test_settings.yaml"))


@pytest.fixture
def teardown():
    yield
    files = os.listdir(settings["download_directory"])
    for file in files:
        os.remove(settings["download_directory"] + "/" + file)


@pytest.mark.skipif(settings["enable_gdrive_source"] is False,
                    reason="enable_gdrive_source disabled "
                           "in test settings..skipping")
def test_gdrive_list_files():
    gdrive = GDriveClient(settings=settings)
    files = gdrive.list(f"'{settings['gdrive_directory']}' in parents")
    assert files != []


@pytest.mark.skipif(settings["enable_gdrive_source"] is False,
                    reason="enable_gdrive_source disabled "
                           "in test settings..skipping")
def test_next_page_token():
    gdrive = GDriveClient(settings=settings)
    files = gdrive.list(query="'1R0HgPqj_4OVKYk2gcIoOhqFzeWxc7r5p' in parents",
                        page_size=2)
    assert len(files) > 2


@pytest.mark.skipif(settings["enable_gdrive_source"] is False,
                    reason="enable_gdrive_source disabled "
                           "in test settings..skipping")
def test_file_download(teardown):
    gdrive = GDriveClient(settings=settings)
    files = gdrive.list(f"'{settings['gdrive_directory']}' in parents")
    file_id, filename = files[0]["id"], files[0]["name"]
    gdrive.download(file_id, filename)
    downloads = [file for file in os.listdir(settings["download_directory"])]
    assert filename in downloads
