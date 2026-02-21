from utils.data_loader import get_all_tracks, load_track

DATA_PATH = "/mnt/c/Users/Vikram Kumar/lpr_project/data/raw/wYe7pBJ7-train/train"

tracks = get_all_tracks(DATA_PATH)

print("Total tracks:", len(tracks))

sample = load_track(tracks[0])
print(sample)