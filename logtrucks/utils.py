import re


def is_lp_valid(lp):
    lp_regex = r"[ABCEHKMOPTXY]{1,2}\s\d{4}\s[ABCEHKMOPTXY]{2}"
    res = re.match(lp_regex, lp)
    if res:
        return True
    return False


def get_uuid_from_filename(filename):
    uuid_regex = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    for part in filename.split("_"):
        matched = re.search(uuid_regex, part)
        if matched:
            return matched.group(0)
