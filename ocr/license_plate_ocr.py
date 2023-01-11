import os
from collections import OrderedDict
import torch
import yaml

from ocr.dataset import AlignCollate
from ocr.model import Model
from ocr.utils import CTCLabelConverter, AttrDict


package_directory = os.path.dirname(os.path.abspath(__file__))


class OCRecognizer:

    def __init__(self, model_name):
        self.path = os.path.join(package_directory, "weights", model_name)
        self.opt = self.get_config()
        self.converter = CTCLabelConverter(self.opt.character_list)
        self.device = self.get_device()

    def get_config(self):
        with open(self.path + ".yaml", "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        opt = AttrDict(opt)
        return opt

    def get_device(self):
        cuda = False
        if torch.cuda.is_available():
            cuda = True
        return torch.device("cuda:0" if cuda else "cpu")

    def load_model(self):
        self.opt.num_class = len(self.converter.character)
        model = Model(self.opt)
        original_dict = torch.load(self.path + ".pth",
                                   map_location=self.device)
        new_dict = OrderedDict()
        for k, v in original_dict.items():
            new_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_dict)
        return model

    def predict(self, image, model):
        align_collate = AlignCollate(imgH=self.opt.imgH,
                                     imgW=self.opt.imgW,
                                     keep_ratio_with_pad=True,
                                     contrast_adjust=0)
        image_tensors, _ = align_collate([(image, None)])
        batch_size = image_tensors.size(0)
        text_for_pred = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)
        preds = model(image_tensors, text_for_pred)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = self.converter.decode_greedy(preds_index.data,
                                                 preds_size.data)
        return preds_str
