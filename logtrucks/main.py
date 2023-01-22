import re
import os
import statistics
import torch
import yaml

from logtrucks.iwpod_net.detector import iwpod_detect
from logtrucks.yolo.detect import YoloDetector
from logtrucks.ocr.license_plate_ocr import OCRecognizer
from logtrucks.utils import is_lp_valid, get_uuid_from_filename


class DetectLogic:

    def __init__(self, settings="settings.yaml"):
        self.settings = yaml.safe_load(open(settings))

    def get_new_inputs(self):
        pass

    def start_detections(self, source):
        self.yolo_detect(source)
        images_with_lps = iwpod_detect(self.settings["results_dir"])
        if images_with_lps:
            self.ocr(images_with_lps)
        self.reduce_detected_images()

    def yolo_detect(self, source, gdrive_id=None):
        with torch.no_grad():
            detector = YoloDetector(source=source,
                                    weights=self.settings["yolo_weights"],
                                    save_dir=self.settings["results_dir"],
                                    gdrive_id=gdrive_id)
            detector.detect(conf_thres=0.45)

    def ocr(self, images_with_lps):
        ocr = OCRecognizer(self.settings)
        ocr.read(images_with_lps)

    def reduce_detected_images(self):
        files = [file for file in os.listdir(self.settings["results_dir"])
                 if not file.endswith(".lpr.jpg")]
        uuids = [get_uuid_from_filename(file) for file in files]
        frames_dict = dict.fromkeys(uuids, None)
        for key in frames_dict.keys():
            frames_dict[key] = [re.findall(r"_(.*?)\.", file)[0]
                                for file in files
                                if get_uuid_from_filename(file) == key]
        for uuid in frames_dict:
            indices = [int(index) for index in frames_dict[uuid]]
            files_to_keep = [uuid + "_" + str(frame) for frame in
                             [min(indices), max(indices),
                              int(statistics.median(indices))]]
            files_to_delete = [file for file in
                               os.listdir(self.settings["results_dir"])
                               if file.split(".")[0] not in files_to_keep
                               and get_uuid_from_filename(file) == uuid
                               and not file.endswith(".lpr.jpg")]
            for f in files_to_delete:
                os.remove(self.settings["results_dir"] + "/" + f)
