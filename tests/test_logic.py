import pytest
from main import DetectLogic


def test_lp_validation():
    valid = ["CA 3812 MA", "C 3812 HH"]
    invalid = ["AZ 3812 HH", "A 3300 M", "T 338 AM"]
    for lp in valid:
        assert DetectLogic.is_lp_valid(lp) is True
    for lp in invalid:
        assert DetectLogic.is_lp_valid(lp) is not True




def test_reduce_detected_images():
    pass