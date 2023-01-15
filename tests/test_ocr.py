import os
import yaml

from PIL import Image

from logtrucks.ocr.license_plate_ocr import OCRecognizer


settings = yaml.safe_load(open("logtrucks/settings.yaml"))


def test_ocr_predictions():
    ocr = OCRecognizer(settings["ocr_weights"])
    model = ocr.load_model()
    image_path = "tests/assets/lps/"
    for image in os.listdir(image_path):
        im = Image.open(image_path + image)
        prediction = ocr.predict(im, model)
        ground_truth = image.split("_")[1]
        assert prediction[0].replace(" ", "") == ground_truth
