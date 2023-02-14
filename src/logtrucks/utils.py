import os
import re
import statistics
import yaml

from typing import Union


def is_lp_valid(lp: str) -> bool:
    lp_regex = r"[ABCEHKMOPTXY]{1,2}\s\d{4}\s[ABCEHKMOPTXY]{2}"
    res = re.match(lp_regex, lp)
    if res:
        return True
    return False


def get_uuid_from_filename(filename: str) -> str:
    uuid_regex = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    for part in filename.split("_"):
        matched = re.search(uuid_regex, part)
        if matched:
            return matched.group(0)


def reduce_detected_images(results_dir: str) -> None:
    files = [file for file in os.listdir(results_dir)
             if not file.endswith(".lpr.jpg")]
    uuids = [uuid for file in files
             if (uuid := get_uuid_from_filename(file)) is not None]
    frames_dict = dict.fromkeys(uuids, None)
    for key in frames_dict.keys():
        frames_dict[key] = [re.findall(r"_(.*?)\.", file)[0]
                            for file in files
                            if get_uuid_from_filename(file) == key]
    for uuid in frames_dict:
        indices = [int(index) for index in frames_dict[uuid]]
        files_to_keep = [uuid + "_" + str(frame) for frame in
                         [min(indices), max(indices),
                          int(statistics.median(indices))]]
        files_to_delete = [file for file in
                           os.listdir(results_dir)
                           if file.split(".")[0] not in files_to_keep
                           and get_uuid_from_filename(file) == uuid
                           and not file.endswith(".lpr.jpg")]
        for f in files_to_delete:
            os.remove(results_dir + "/" + f)


def load_settings(settings: Union[str, dict]) -> dict:
    if type(settings) == str:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        settings = yaml.safe_load(open(os.path.join(current_dir, settings)))
    return settings
