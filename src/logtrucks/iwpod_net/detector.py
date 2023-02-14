import os

import cv2

from src.logtrucks.iwpod_net.src.keras_utils import load_model
from src.logtrucks.iwpod_net.src.keras_utils import detect_lp_width
from src.logtrucks.iwpod_net.src.utils import im2single


package_directory = os.path.dirname(os.path.abspath(__file__))


def iwpod_detect(image_dir: str, conf_thres: float = 0.35) -> list:
    files = [file for file in os.listdir(image_dir) if file.endswith("jpg")
             or file.endswith("jpeg")]
    detection_in_images = list()
    for image in files:
        ocr_input_size = [80, 240]
        iwpod_net = load_model(os.path.join(package_directory,
                                            "weights", "iwpod_net"))
        input_vehicle = cv2.imread(image_dir + "/" + image)

        aspect_ration = 1
        wpod_resolution = 640  # larger if full image is used directly
        lp_output_resolution = tuple(ocr_input_size[::-1])
        detections, images, _ = detect_lp_width(iwpod_net,
                                                im2single(input_vehicle),
                                                wpod_resolution*aspect_ration,
                                                2**4,
                                                lp_output_resolution,
                                                conf_thres)
        if detections:
            d_id = image.split(".")[0]
            for i, img in enumerate(images):
                filename = f"{image_dir}/{d_id}_{str(i)}.lpr.jpg"
                cv2.imwrite(filename, img * 255)
                detection_in_images.append(filename)

    return detection_in_images
