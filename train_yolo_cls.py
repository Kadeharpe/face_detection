import os
import shutil
import random
from collections import defaultdict

IMG_DIR = r"img_align_celeba\img_align_celeba"
IDENTITY_FILE = r"identity_CelebA.txt"
OUT_DIR = r"celeba_yolo_cls"

# how many identities to use for quick demo
MAX_IDENTITIES = 10
MIN_IMAGES_PER_ID = 20

random.seed(42)

# -----------------------------
# READ IDENTITY LABELS
# -----------------------------
id_to_images = defaultdict(list)

with open(IDENTITY_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        img_name, identity = parts
        id_to_images[identity].append(img_name)

# keep only identities with enough images
filtered = {k: v for k, v in id_to_images.items() if len(v) >= MIN_IMAGES_PER_ID}

# choose a small subset
selected_ids = list(filtered.keys())[:MAX_IDENTITIES]

# recreate output folders
for split in ["train", "val", "test"]:
    for identity in selected_ids:
        os.makedirs(os.path.join(OUT_DIR, split, identity), exist_ok=True)

# split and copy images
for identity in selected_ids:
    imgs = filtered[identity][:]
    random.shuffle(imgs)

    n = len(imgs)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_imgs = imgs[:train_end]
    val_imgs = imgs[train_end:val_end]
    test_imgs = imgs[val_end:]

    for split_name, split_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:
        for img_name in split_imgs:
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(OUT_DIR, split_name, identity, img_name)
            if os.path.exists(src):
                shutil.copy(src, dst)

print("Done creating classification dataset.")
print(f"Selected identities: {len(selected_ids)}")

from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="celeba_yolo_cls",
    epochs=3,      # keep small for speed
    imgsz=224,
    batch=16
)

metrics = model.val()
print(metrics)