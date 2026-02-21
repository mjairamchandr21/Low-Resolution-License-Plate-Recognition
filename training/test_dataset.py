from training.dataset import LPRDataset

DATA_PATH = "/mnt/c/Users/Vikram Kumar/lpr_project/data/raw/wYe7pBJ7-train/train"

dataset = LPRDataset(DATA_PATH)

print("Total samples:", len(dataset))

img, label, length = dataset[0]

print("Image shape:", img.shape)
print("Label:", label)
print("Length:", length)