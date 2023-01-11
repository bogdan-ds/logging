import re
import os
import statistics
import yaml

from PIL import Image
from sqlalchemy.orm import sessionmaker

from schema import Trucks, engine
from iwpod_net.detector import iwpod_detect
from yolo.detect import YoloDetector
from ocr.license_plate_ocr import OCRecognizer


class DetectLogic:

    def __init__(self):
        session = sessionmaker(bind=engine)
        self.session = session()
        self.settings = yaml.safe_load(open("settings.yaml"))

    def get_new_inputs(self):
        pass

    def start_detections(self, source):
        detector = YoloDetector(source=source,
                                weights=self.settings["yolo_weights"])
        detector.detect(conf_thres=0.45)
        images_with_lps = iwpod_detect(self.settings["results_dir"])
        if images_with_lps:
            ocr = OCRecognizer(self.settings["ocr_weights"])
            model = ocr.load_model()
            for image in images_with_lps:
                im = Image.open(image)
                prediction = ocr.predict(im, model)
                lp_valid = self.is_lp_valid(prediction)
        # TODO reduce images, write to db, write field if LP is valid or not

    def reduce_detected_images(self):
        files = [file for file in os.listdir(self.settings["results_dir"])
                 if file.endswith("jpg") or file.endswith("jpeg")]
        uuids = [file.split("_")[0] for file in files]
        frames_dict = dict.fromkeys(uuids, None)
        for key in frames_dict.keys():
            frames_dict[key] = [re.findall(r"_(.*?)\.", file)[0]
                                for file in files if file.split("_")[0] == key]
        for uuid in frames_dict:
            files_to_keep = [uuid + "_" + frame for frame in
                             [min(frames_dict[uuid]), max(frames_dict[uuid]),
                              statistics.median(frames_dict[uuid])]]
            files_to_delete = [file for file in
                               os.listdir(self.settings["results_dir"])
                               if file.split(".")[0] not in files_to_keep
                               and file.split("_")[0] == uuid]
            for f in files_to_delete:
                os.remove(self.settings["results_dir"] + "/" + f)

    @staticmethod
    def is_lp_valid(lp):
        lp_regex = r"[ABCEHKMOPTXY]{1,2}\s\d{4}\s[ABCEHKMOPTXY]{2}"
        res = re.match(lp_regex, lp)
        if res:
            return True
        return False
