from models.ocr_model import EasyOCRModel
from utils.data_loader import get_all_tracks, load_track

DATA_PATH = "/mnt/c/Users/Vikram Kumar/lpr_project/data/raw/wYe7pBJ7-train/train"

tracks = get_all_tracks(DATA_PATH)

model = EasyOCRModel(device="cpu")

sample = load_track(tracks[0])

print("GT:", sample["plate_text"])

for img in sample["images"]:
    pred, conf = model.predict(img)
    print("Pred:", pred, "Conf:", conf)