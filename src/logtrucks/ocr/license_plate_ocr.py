import os
from collections import OrderedDict
import torch
import yaml

from PIL import Image

from sqlalchemy.orm import sessionmaker

from src.logtrucks.schema import Detections, engine
from src.logtrucks.utils import get_uuid_from_filename, is_lp_valid
from src.logtrucks.ocr.dataset import AlignCollate
from src.logtrucks.ocr.model import Model
from src.logtrucks.ocr.utils import CTCLabelConverter, AttrDict


package_directory = os.path.dirname(os.path.abspath(__file__))


class OCRecognizer:

    def __init__(self, settings: dict):
        self.settings = settings
        self.path = os.path.join(package_directory, "weights",
                                 settings["ocr_weights"])
        self.opt = self.get_config()
        self.converter = CTCLabelConverter(self.opt.character_list)
        self.device = self.get_device()

    def get_config(self) -> AttrDict:
        with open(self.path + ".yaml", "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        opt = AttrDict(opt)
        return opt

    def get_device(self) -> torch.device:
        cuda = False
        if torch.cuda.is_available():
            cuda = True
        return torch.device("cuda:0" if cuda else "cpu")

    def read(self, images_with_lps: list) -> list:
        model = self.load_model()
        results = list()
        for image in images_with_lps:
            prediction = self.predict(image, model)
            lp_valid = is_lp_valid(prediction[0])
            if not lp_valid:
                os.remove(image)
            else:
                lp = prediction[0].replace(" ", "")
                if self.settings["rename_lp_image"]:
                    os.rename(image,
                              f'{self.settings["results_dir"]}/_'
                              f'{lp}_.lpr.jpg')
                uuid = get_uuid_from_filename(image)
                if self.settings["write_to_db"]:
                    self.update_db_with_lp(uuid, lp)
                results.append(lp)
        return results

    def load_model(self) -> Model:
        self.opt.num_class = len(self.converter.character)
        model = Model(self.opt)
        original_dict = torch.load(self.path + ".pth",
                                   map_location=self.device)
        new_dict = OrderedDict()
        for k, v in original_dict.items():
            new_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_dict)
        return model

    def predict(self, image: str, model: Model) -> str:
        align_collate = AlignCollate(imgH=self.opt.imgH,
                                     imgW=self.opt.imgW,
                                     keep_ratio_with_pad=True,
                                     contrast_adjust=0)
        im = Image.open(image)
        image_tensors, _ = align_collate([(im, None)])
        batch_size = image_tensors.size(0)
        text_for_prediction = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)
        predictions = model(image_tensors, text_for_prediction)
        predictions_size = torch.IntTensor([predictions.size(1)] * batch_size)
        _, predictions_index = predictions.max(2)
        predictions_index = predictions_index.view(-1)
        prediction_string = self.converter.decode_greedy(predictions_index.data,
                                                         predictions_size.data)
        return prediction_string

    def update_db_with_lp(self, uuid: str, lp_prediction: str) -> None:
        session = sessionmaker(bind=engine)
        session = session()
        session.query(Detections).filter(
            Detections.id == uuid).update({
                "license_plate_prediction": lp_prediction})
        session.commit()
