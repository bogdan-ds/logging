import os

import cv2

from iwpod_net.src.keras_utils import load_model
from iwpod_net.src.keras_utils import detect_lp_width
from iwpod_net.src.utils import im2single


package_directory = os.path.dirname(os.path.abspath(__file__))


def iwpod_detect(image_dir, conf_thres=0.35):
    files = [file for file in os.listdir(image_dir) if file.endswith("jpg")
             or file.endswith("jpeg")]
    detection_in_images = list()
    for image in files:
        ocr_input_size = [80, 240]
        iwpod_net = load_model(os.path.join(package_directory,
                                            "weights", "iwpod_net"))
        Ivehicle = cv2.imread(image_dir + "/" + image)

        ASPECTRATIO = 1
        WPODResolution = 640  # larger if full image is used directly
        lp_output_resolution = tuple(ocr_input_size[::-1])
        detections, images, _ = detect_lp_width(iwpod_net,
                                                im2single(Ivehicle),
                                                WPODResolution*ASPECTRATIO,
                                                2**4,
                                                lp_output_resolution,
                                                conf_thres)
        if detections:
            d_id = image.split("_")[0]
            for i, img in enumerate(images):
                filename = f"{image_dir}/{d_id}_{str(i)}.lpr.jpg"
                cv2.imwrite(filename, img * 255)
                detection_in_images.append(filename)

    return detection_in_images
