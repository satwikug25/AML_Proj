import os
import random
import csv
import requests
from urllib.parse import quote
from shutil import copy2

# -----------------------------
# CONFIG
# -----------------------------

# 20 classes
MEDICINES = [
    "ibuprofen",
    "acetaminophen",
    "naproxen",
    "cetirizine",
    "loratadine",
    "amoxicillin",
    "azithromycin",
    "omeprazole",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "simvastatin",
    "aspirin",
    "clopidogrel",
    "amlodipine",
    "losartan",
    "pantoprazole",
    "furosemide",
    "levothyroxine",
    "albuterol"
]

# Where to store raw downloaded images (per class)
RAW_ROOT = "medicine_boxes_raw"

# Final split dataset: train/val/test subdirs
SPLIT_ROOT = "medicine_boxes_split"

# Train/val/test ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15   # test = 1 - train - val

# Per-class limits (you can tweak)
PAGES_PER_DRUG = 3          # more pages -> more SPL entries
PAGESIZE = 25               # SPL entries per page
MAX_IMAGES_PER_DRUG = 150   # cap to avoid insane counts

# Random seed for reproducible splits
RANDOM_SEED = 42


# -----------------------------
# UTILITIES
# -----------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "course-project/1.0 (contact: your_email@example.com)"})

def get_json(url, params=None):
    """Fetch JSON with basic error handling."""
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# -----------------------------
# SCRAPING FROM DAILYMED
# -----------------------------

def scrape_medicine_images(drug_name, pages=PAGES_PER_DRUG, pagesize=PAGESIZE,
                           max_images_per_drug=MAX_IMAGES_PER_DRUG):
    """
    Scrape images from DailyMed for a given drug_name.
    Uses the official JSON API:
      - /services/v2/spls.json to get SPL entries (setid)
      - /services/v2/spls/{setid}/media.json to get image URLs
    """
    print(f"\n🔍 Collecting images for: {drug_name}")
    class_folder = os.path.join(RAW_ROOT, drug_name.replace(" ", "_"))
    os.makedirs(class_folder, exist_ok=True)

    base_spls_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

    downloaded = 0
    seen_setids = set()

    for page in range(1, pages + 1):
        if downloaded >= max_images_per_drug:
            break

        params = {
            "drug_name": drug_name,
            "pagesize": pagesize,
            "page": page,
        }

        try:
            data = get_json(base_spls_url, params=params)
        except Exception as e:
            print(f"  [WARN] Failed to fetch SPL list for {drug_name} page {page}: {e}")
            continue

        spls = data.get("data", [])
        if not spls:
            print(f"  No SPLs on page {page} for {drug_name}")
            break

        for spl in spls:
            if downloaded >= max_images_per_drug:
                break

            setid = spl.get("setid")
            if not setid or setid in seen_setids:
                continue
            seen_setids.add(setid)

            media_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/media.json"
            try:
                media_data = get_json(media_url)
            except Exception as e:
                print(f"  [WARN] Failed to fetch media for setid={setid}: {e}")
                continue

            media_items = media_data.get("data", {}).get("media", [])
            if not media_items:
                continue

            for m in media_items:
                if downloaded >= max_images_per_drug:
                    break

                mime = m.get("mime_type", "")
                url = m.get("url")

                # Only real images
                if not url or not mime.startswith("image/"):
                    continue

                try:
                    img_resp = SESSION.get(url, timeout=15)
                    img_resp.raise_for_status()
                    ext = mime.split("/")[-1]  # e.g., "jpeg", "png"
                    ext = "jpg" if ext.lower() in ["jpeg", "pjpeg"] else ext

                    fname = f"{drug_name.replace(' ', '_')}_{setid}_{downloaded:04d}.{ext}"
                    path = os.path.join(class_folder, fname)
                    with open(path, "wb") as f:
                        f.write(img_resp.content)
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"  Downloaded {downloaded} images for {drug_name}...")
                except Exception as e:
                    print(f"    [WARN] Failed downloading image {url}: {e}")
                    continue

    print(f"✅ Done: {drug_name} → {downloaded} images in {class_folder}")


# -----------------------------
# TRAIN / VAL / TEST SPLITS
# -----------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def split_dataset():
    """
    Takes images in RAW_ROOT/{class} and splits into:
      SPLIT_ROOT/{train,val,test}/{class}/
    Also writes a CSV 'dataset_index.csv' with columns:
      filename, class, split
    """
    print("\n📂 Creating train/val/test split...")

    random.seed(RANDOM_SEED)

    splits = ["train", "val", "test"]
    split_root = SPLIT_ROOT

    for split in splits:
        ensure_dir(os.path.join(split_root, split))

    index_rows = [("filename", "class", "split")]

    classes = [d for d in os.listdir(RAW_ROOT)
               if os.path.isdir(os.path.join(RAW_ROOT, d))]

    for cls in sorted(classes):
        class_raw_dir = os.path.join(RAW_ROOT, cls)
        imgs = [f for f in os.listdir(class_raw_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not imgs:
            print(f"  [WARN] No images found for class '{cls}', skipping.")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        # Ensure at least 1 image in train; adjust if very small class
        if n_train == 0 and n > 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_test = max(n_test - 1, 0)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        print(f"Class '{cls}': total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

        for split_name, split_list in zip(splits, [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(split_root, split_name, cls)
            ensure_dir(split_class_dir)

            for img_name in split_list:
                src = os.path.join(class_raw_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                copy2(src, dst)

                rel_path = os.path.join(split_name, cls, img_name)
                index_rows.append((rel_path, cls, split_name))

    # write CSV index
    csv_path = os.path.join(split_root, "dataset_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(index_rows)

    print(f"\n✅ Splitting complete! Dataset stored in '{split_root}'")
    print(f"   Index CSV: {csv_path}")


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    os.makedirs(RAW_ROOT, exist_ok=True)
    os.makedirs(SPLIT_ROOT, exist_ok=True)

    print("=== STEP 1: SCRAPING IMAGES FROM DAILYMED ===")
    for med in MEDICINES:
        scrape_medicine_images(med)

    print("\n=== STEP 2: CREATING TRAIN/VAL/TEST SPLITS ===")
    split_dataset()

    print("\n🎉 All done!")
