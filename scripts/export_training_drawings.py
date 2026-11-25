"""Export example drawings from the synthetic training set (same seeds as scripts/train_segnet.py).

    python scripts/export_training_drawings.py [n_train] [n_unseen]

Writes to samples/training_drawings/:
  train_seedNNNN_<ppm>ppm.png    exact drawings the learned reader was trained on (seeds 0-1999)
  unseen_seedNNNN_<ppm>ppm.png   drawings from the same generator the model has never seen (good for demos)
  *_labels.png                   the answer key the model learns: equipment height, hazard grade, leak points
  overview.png                   all drawings on one sheet
The <ppm> in each file name is the scale (pixels per metre) to enter when uploading it in step 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectorplace import scene  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples" / "training_drawings"
EQ_COL = {0: (245, 245, 245), 1: (190, 190, 190), 2: (125, 125, 125), 3: (60, 60, 60)}      # ground, 2/5/9 m
ZONE_COL = {1: (255, 240, 160), 2: (255, 205, 150), 3: (250, 160, 160)}                     # grade C/B/A


def generate(seed):
    """Identical to make() in scripts/train_segnet.py up to the render step."""
    rng = np.random.default_rng(10_000 + seed)
    spec, style = scene.random_spec(rng), scene.random_style(rng)
    img, lab = scene.render(spec, style, rng)
    return img, lab, style.px_per_m


def label_image(img, lab, ppm):
    eq, zn = lab["equip"], lab["zone"]
    out = np.full(img.shape, 245, np.uint8)
    for g, c in ZONE_COL.items():
        out[zn == g] = c
    for k, c in EQ_COL.items():
        if k:
            out[eq == k] = c
    for x, y in lab["leaks"]:
        cv2.circle(out, (int(x * ppm), int(y * ppm)), max(4, int(0.6 * ppm)), (30, 80, 220), -1)
    return out


def save(path, rgb):
    cv2.imwrite(str(path), rgb[..., ::-1])


def main(n_train=8, n_unseen=4):
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for kind, seeds in (("train", range(n_train)), ("unseen", range(50_000, 50_000 + n_unseen))):
        for s in seeds:
            img, lab, ppm = generate(s)
            name = f"{kind}_seed{s:04d}_{ppm:g}ppm"
            save(OUT / f"{name}.png", img)
            save(OUT / f"{name}_labels.png", label_image(img, lab, ppm))
            t = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            t = cv2.copyMakeBorder(t, 26, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"{kind} seed {s} ({ppm:g} px/m)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (20, 20, 20) if kind == "train" else (200, 40, 40), 1, cv2.LINE_AA)
            tiles.append(t)
            print(f"{name}: {img.shape[1]}x{img.shape[0]} px, {len(lab['leaks'])} leak points")
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows = [np.hstack(tiles[i:i + 4] + [np.full_like(tiles[0], 255)] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    save(OUT / "overview.png", np.vstack(rows))
    print("written to", OUT)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
