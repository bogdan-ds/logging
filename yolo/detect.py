import os
import time
from uuid import uuid4
from pathlib import Path

import cv2
import torch
from numpy import random

from yolo.utils.datasets import LoadImages
from yolo.utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from yolo.utils.plots import plot_one_box
from yolo.utils.torch_utils import select_device, time_synchronized


package_directory = os.path.dirname(os.path.abspath(__file__))


class YoloDetector:

    def __init__(self, source, weights,
                 device="", save_dir="res",
                 detection_max_frame_length=50, detection_frame_buffer=5):
        self.source = source
        self.weights = os.path.join(package_directory, "weights", weights)
        self.device = device
        self.save_dir = save_dir
        self.detection_max_frame_length = detection_max_frame_length
        self.detection_frame_buffer = detection_frame_buffer
        self.detection_start_frame = 0
        self.detection_last_frame = 0
        self.current_detection_id = None

    def detect(self,
               imgsz=640,
               conf_thres=0.25,
               iou_thres=0.45):

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

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
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
                _, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:],
                                              det[:, :4],
                                              im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=1)
                    self._detect_logic(im0,
                                       frame)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) '
                      f'Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        print(f'Done. ({time.time() - t0:.3f}s)')

    def _detect_logic(self, image, frame):
        if not self.detection_last_frame:
            self._reset_detection_values(frame)
            print(f"No last frame, setting values: "
                  f"{str(self.current_detection_id)}")
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
            print(f"Detection after max length, setting values: "
                  f"{str(self.current_detection_id)}, "
                  f"last frame: {self.detection_last_frame}")

    def _reset_detection_values(self, frame):
        self.current_detection_id = uuid4()
        self.detection_start_frame = frame
        self.detection_last_frame = frame

    def _save_frame(self, image, frame):
        self.detection_last_frame = frame
        save_path = self.save_dir + "/" + str(
            self.current_detection_id) + "_" + str(frame) + ".jpg"
        cv2.imwrite(save_path, image)
        print(f"Saving frame to {save_path}")


if __name__ == '__main__':

    with torch.no_grad():
        detector = YoloDetector(source="/home/bogdan/Videos/logging2.mp4",
                                weights="very-good-best.pt")
        detector.detect(conf_thres=0.40)
