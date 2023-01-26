import yaml

from sqlalchemy.orm import sessionmaker

from logtrucks.gdrive_client import GDriveClient
from logtrucks.schema import Detections, engine
from logtrucks.utils import reduce_detected_images
from logtrucks.iwpod_net.detector import iwpod_detect
from logtrucks.yolo.detect import YoloDetector
from logtrucks.ocr.license_plate_ocr import OCRecognizer


class DetectLogic:

    def __init__(self, settings="settings.yaml"):
        self.settings = yaml.safe_load(open(settings))

    def start_detections(self, source):
        self.yolo_detect(source)
        images_with_lps = iwpod_detect(self.settings["results_dir"])
        if images_with_lps:
            self.ocr(images_with_lps)
        reduce_detected_images(self.settings["results_dir"])

    def yolo_detect(self, source, gdrive_id=None):
        detector = YoloDetector(source=source,
                                settings=self.settings,
                                gdrive_id=gdrive_id)
        detector.detect(conf_thres=0.45)

    def ocr(self, images_with_lps):
        ocr = OCRecognizer(self.settings)
        ocr.read(images_with_lps)


class GDriveDetectLogic(DetectLogic):

    def __init__(self, settings):
        super().__init__(settings)
        self.client = GDriveClient(settings=self.settings)

    def process_new_items(self):
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

    def get_processed_gdrive_ids(self):
        session = sessionmaker(bind=engine)
        session = session()
        result = session.query(Detections.gdrive_file_id)
        return set([gdrive_id[0] for gdrive_id in result.all()])

    def get_current_gdrive_items(self):
        query = f"'{self.settings['gdrive_directory']}' in parents " \
                f"and (mimeType contains 'image/' or " \
                f"mimeType contains 'video/')"
        files = self.client.list(query=query)
        return {file["id"]: file["name"] for file in files}
