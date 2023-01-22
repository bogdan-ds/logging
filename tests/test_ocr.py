import os
import yaml

from logtrucks.ocr.license_plate_ocr import OCRecognizer


settings = yaml.safe_load(open("tests/test_settings.yaml"))


def test_ocr_predictions():
    settings["write_to_db"] = False
    settings["results_dir"] = "tests/assets/lps/"
    settings["rename_lp_image"] = False
    ocr = OCRecognizer(settings)
    image_path = settings["results_dir"]
    images_with_lps = [image_path + image for image in os.listdir(image_path)]
    results = ocr.read(images_with_lps)
    ground_truth = [file.split("_")[1]
                    for file in os.listdir("tests/assets/lps")]
    assert results.sort() == ground_truth.sort()
