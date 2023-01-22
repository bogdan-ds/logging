from uuid import uuid4

from logtrucks.utils import is_lp_valid, get_uuid_from_filename


def test_lp_validation():
    valid = ["CA 3812 MA", "C 3812 HH"]
    invalid = ["AZ 3812 HH", "A 3300 M", "T 338 AM"]
    for lp in valid:
        assert is_lp_valid(lp) is True
    for lp in invalid:
        assert is_lp_valid(lp) is not True


def test_get_uuid_from_filename():
    uuid = str(uuid4())
    filename = f"{uuid}_128.jpg"
    assert get_uuid_from_filename(filename) == uuid
