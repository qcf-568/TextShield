import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Input image dir.")
parser.add_argument("--output", required=True, type=str, help="Output image dir.")
args = parser.parse_args()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def resize_to_mult_28(in_dir: str, out_dir: str = None, overwrite: bool = False, interp=cv2.INTER_AREA):
    in_dir = os.path.abspath(in_dir)
    out_dir = in_dir if (overwrite or out_dir is None) else os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for root, _, names in os.walk(in_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXT:
                files.append(os.path.join(root, n))

    for p in tqdm(files, desc="Resizing", unit="img"):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        nh = max(28, int(round(h / 28)) * 28)
        nw = max(28, int(round(w / 28)) * 28)
        if (nh, nw) != (h, w):
            img = cv2.resize(img, (nw, nh), interpolation=interp)

        rel = os.path.relpath(p, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

if __name__ == "__main__":
    resize_to_mult_28(args.input, out_dir=args.output)  # 或 overwrite=True
