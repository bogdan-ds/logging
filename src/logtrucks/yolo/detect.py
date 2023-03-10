import datetime
import os
import time
from uuid import uuid4
from pathlib import Path

import cv2
import torch
import numpy as np
# from numpy import random
from sqlalchemy.orm import sessionmaker

from src.logtrucks.schema import Detections, engine
from src.logtrucks.yolo.utils.datasets import LoadImages
from src.logtrucks.yolo.utils.general import check_img_size, \
    non_max_suppression, scale_coords, set_logging
# from src.logtrucks.yolo.utils.plots import plot_one_box
from src.logtrucks.yolo.utils.torch_utils import select_device, \
    time_synchronized


package_directory = os.path.dirname(os.path.abspath(__file__))


class YoloDetector:

    def __init__(self, source: str, settings: dict,
                 device: str = "", gdrive_id: str = None):
        self.source = source
        self.settings = settings
        self.weights = os.path.join(package_directory, "weights",
                                    settings["yolo_weights"])
        self.device = device
        self.save_dir = settings["results_dir"]
        self.detection_max_frame_length = settings["detection_max_frame_length"]
        self.detection_frame_buffer = settings["detection_frame_buffer"]
        self.gdrive_id = gdrive_id
        self.detection_start_frame = 0
        self.detection_last_frame = 0
        self.current_detection_id = None
        self.cap = None

    def detect(self,
               imgsz: int = 640,
               conf_thres: float = 0.25,
               iou_thres: float = 0.45) -> str:

        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # load FP32 model
        ckpt = torch.load(self.weights, map_location=device)
        model = ckpt['ema' if ckpt.get('ema') else 'model']\
            .float().fuse().eval()
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16

        # Set Dataloader
        dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
        self.cap = dataset.cap

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        with torch.no_grad():
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or
                                             old_img_h != img.shape[2] or
                                             old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img)[0]

                # Inference
                t1 = time_synchronized()
                pred = model(img)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                t3 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    _, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:],
                                                  det[:, :4],
                                                  im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}, "  # add to string

                        # Write results to image;
                        # maybe not a good idea for LP detection
                        # for *xyxy, conf, cls in reversed(det):
                        #    # Add bbox to image
                        #    label = f'{names[int(cls)]} {conf:.2f}'
                        #    plot_one_box(xyxy, im0, label=label,
                        #                 color=colors[int(cls)], line_thickness=1)
                        self._detect_logic(im0, frame)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) '
                          f'Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            print(f'Done. ({time.time() - t0:.3f}s)')
        return s

    def _detect_logic(self, image: np.ndarray, frame: int) -> None:
        if not self.detection_last_frame:
            self._reset_detection_values(frame)
            self._save_frame(image, frame)
            if self.settings["write_to_db"]:
                self._save_to_db(self.source, frame,
                                 self.current_detection_id)
        elif frame == self.detection_start_frame:
            pass
        elif frame != self.detection_start_frame and \
                frame < (self.detection_start_frame +
                         self.detection_frame_buffer):
            self._save_frame(image, frame)
        elif frame == (self.detection_last_frame + 1):
            self._save_frame(image, frame)
        elif frame > (self.detection_last_frame +
                      self.detection_max_frame_length):
            self._reset_detection_values(frame)

    def _reset_detection_values(self, frame: int) -> None:
        self.current_detection_id = str(uuid4())
        self.detection_start_frame = frame
        self.detection_last_frame = frame

    def _save_frame(self, image: np.ndarray, frame: int) -> None:
        self.detection_last_frame = frame
        save_path = self.save_dir + "/" + str(
            self.current_detection_id) + "_" + str(frame) + ".jpg"
        cv2.imwrite(save_path, image)

    def _save_to_db(self, filename: str, frame: int, uuid: str) -> None:
        timestamp = self.get_frame_timestamp(frame) if self.cap else None
        detection = Detections(
            id=uuid,
            ingestion_date=datetime.datetime.now(),
            source_filename=filename,
            gdrive_file_id=self.gdrive_id,
            time_first_detected=timestamp
        )
        session = sessionmaker(bind=engine)
        session = session()
        session.add(detection)
        session.commit()

    def get_frame_timestamp(self, frame: int) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return frame / fps
