import click
from typing import Union

from sqlalchemy.orm import sessionmaker

from src.logtrucks.gdrive_client import GDriveClient
from src.logtrucks.schema import Detections, engine
from src.logtrucks.utils import reduce_detected_images, load_settings
from src.logtrucks.iwpod_net.detector import iwpod_detect
from src.logtrucks.yolo.detect import YoloDetector
from src.logtrucks.ocr.license_plate_ocr import OCRecognizer


class DetectLogic:

    def __init__(self, settings: Union[str, dict] = "settings.yaml"):
        self.settings = load_settings(settings)

    def start_detections(self, source: str) -> None:
        self.yolo_detect(source)
        images_with_lps = iwpod_detect(self.settings["results_dir"])
        if images_with_lps:
            self.ocr(images_with_lps)
        reduce_detected_images(self.settings["results_dir"])

    def yolo_detect(self, source: str, gdrive_id: str = None) -> None:
        detector = YoloDetector(source=source,
                                settings=self.settings,
                                gdrive_id=gdrive_id)
        detector.detect(conf_thres=0.45)

    def ocr(self, images_with_lps: list) -> None:
        ocr = OCRecognizer(self.settings)
        ocr.read(images_with_lps)


class GDriveDetectLogic(DetectLogic):

    def __init__(self, settings: Union[str, dict] = "settings.yaml"):
        super().__init__(settings)
        self.client = GDriveClient(settings=self.settings)

    def process_new_items(self) -> None:
        processed_ids = self.get_processed_gdrive_ids()
        remote_items = self.get_current_gdrive_items()
        remote_ids = set(remote_items.keys())
        new = remote_ids - processed_ids
        if new:
            for file_id in new:
                self.client.download(file_id, remote_items[file_id])
                self.yolo_detect(source=f"{self.settings['download_directory']}"
                                        f"/{remote_items[file_id]}",
                                 gdrive_id=file_id)
            images_with_lps = iwpod_detect(self.settings["results_dir"])
            if images_with_lps:
                self.ocr(images_with_lps)
            reduce_detected_images(self.settings["results_dir"])

    def get_processed_gdrive_ids(self) -> set:
        session = sessionmaker(bind=engine)
        session = session()
        result = session.query(Detections.gdrive_file_id)
        return set([gdrive_id[0] for gdrive_id in result.all()])

    def get_current_gdrive_items(self) -> dict:
        query = f"'{self.settings['gdrive_directory']}' in parents " \
                f"and (mimeType contains 'image/' or " \
                f"mimeType contains 'video/')"
        files = self.client.list(query=query)
        return {file["id"]: file["name"] for file in files}


@click.command()
@click.option(
    "--file",
    "-f",
    help="Path to input file",
    required=True
)
def cli(file: str) -> None:
    detector = DetectLogic()
    detector.start_detections(file)
