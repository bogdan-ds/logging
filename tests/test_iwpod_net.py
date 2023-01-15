import os
import yaml

from logtrucks.iwpod_net.detector import iwpod_detect

settings = yaml.safe_load(open("logtrucks/settings.yaml"))


def test_lp_detection():
    image_dir = "tests/assets/"
    images_with_lps = iwpod_detect(image_dir)
    assert images_with_lps is not []
    for filename in images_with_lps:
        os.remove(filename)
