import re
import os
import statistics
import torch
import yaml

from PIL import Image
from sqlalchemy.orm import sessionmaker

from logtrucks.schema import Trucks, engine
from logtrucks.iwpod_net.detector import iwpod_detect
from logtrucks.yolo.detect import YoloDetector
from logtrucks.ocr.license_plate_ocr import OCRecognizer


class DetectLogic:

    def __init__(self, settings="settings.yaml"):
        session = sessionmaker(bind=engine)
        self.session = session()
        self.settings = yaml.safe_load(open(settings))

    def get_new_inputs(self):
        pass

    def start_detections(self, source):
        with torch.no_grad():
            detector = YoloDetector(source=source,
                                    weights=self.settings["yolo_weights"],
                                    save_dir=self.settings["results_dir"])
            detector.detect(conf_thres=0.45)
        images_with_lps = iwpod_detect(self.settings["results_dir"])
        if images_with_lps:
            ocr = OCRecognizer(self.settings["ocr_weights"])
            model = ocr.load_model()
            for image in images_with_lps:
                im = Image.open(image)
                prediction = ocr.predict(im, model)
                print(f"Predicted LP: {prediction[0]}")
                lp_valid = self.is_lp_valid(prediction[0])
                if not lp_valid:
                    os.remove(image)
                else:
                    os.rename(image,
                              f'{self.settings["results_dir"]}/_'
                              f'{prediction[0].replace(" ", "")}_.lpr.jpg')
        self.reduce_detected_images()
        # TODO write to db

    def reduce_detected_images(self):
        files = [file for file in os.listdir(self.settings["results_dir"])
                 if not file.endswith(".lpr.jpg")]
        uuids = [file.split("_")[0] for file in files]
        frames_dict = dict.fromkeys(uuids, None)
        for key in frames_dict.keys():
            frames_dict[key] = [re.findall(r"_(.*?)\.", file)[0]
                                for file in files if file.split("_")[0] == key]
        for uuid in frames_dict:
            indices = [int(index) for index in frames_dict[uuid]]
            files_to_keep = [uuid + "_" + str(frame) for frame in
                             [min(indices), max(indices),
                              int(statistics.median(indices))]]
            files_to_delete = [file for file in
                               os.listdir(self.settings["results_dir"])
                               if file.split(".")[0] not in files_to_keep
                               and file.split("_")[0] == uuid
                               and not file.endswith(".lpr.jpg")]
            for f in files_to_delete:
                os.remove(self.settings["results_dir"] + "/" + f)

    @staticmethod
    def is_lp_valid(lp):
        lp_regex = r"[ABCEHKMOPTXY]{1,2}\s\d{4}\s[ABCEHKMOPTXY]{2}"
        res = re.match(lp_regex, lp)
        if res:
            return True
        return False
