import os
import json
from typing import List, Dict


def get_all_tracks(data_root: str) -> List[str]:
    """
    Returns list of all track folder paths.
    """
    track_paths = []

    for scenario in os.listdir(data_root):
        scenario_path = os.path.join(data_root, scenario)

        if not os.path.isdir(scenario_path):
            continue

        for layout in os.listdir(scenario_path):
            layout_path = os.path.join(scenario_path, layout)

            if not os.path.isdir(layout_path):
                continue

            for track in os.listdir(layout_path):
                track_path = os.path.join(layout_path, track)

                if os.path.isdir(track_path):
                    track_paths.append(track_path)

    return sorted(track_paths)


def load_track(track_path: str) -> Dict:
    """
    Loads LR image paths and ground truth text for a track.
    """
    images = []
    annotation_path = os.path.join(track_path, "annotations.json")

    for file in os.listdir(track_path):
        if file.startswith("lr-") and file.endswith(".png"):
            images.append(os.path.join(track_path, file))

    images = sorted(images)

    plate_text = None
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            ann = json.load(f)
            plate_text = ann.get("plate_text", None)

    return {
        "track_id": os.path.basename(track_path),
        "images": images,
        "plate_text": plate_text
    }