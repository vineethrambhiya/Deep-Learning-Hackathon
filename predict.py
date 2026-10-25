# data_dir should have the following directory structure:
# data_dir
# - train
## -- 0
## -- 1
# - test


# The test images may also be nested one level deeper as test/test/*.png (this is how the hackathon zip is packaged); both layouts are handled.

# Writes predictions to ./submission.csv with columns ID,Label.

from collections import Counter
from pathlib import Path
import csv

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "best.pt"

# Tuple ordering used below: (spheres, cubes, cylinders).
CLASS0_TUPLES = {
    (0, 0, 4), (0, 1, 3), (0, 2, 2), (0, 3, 1), (0, 4, 0),
    (1, 0, 3), (2, 0, 2), (3, 0, 1), (4, 0, 0),
}
CLASS1_TUPLES = {(1, 1, 2), (1, 3, 0), (3, 1, 0)}

# Tuples that never appeared in the labelled training data, assigned to class 1.
MISSING_TUPLES = {}
MISSING_TUPLES[(1, 2, 1)] = 1
MISSING_TUPLES[(2, 1, 1)] = 1
MISSING_TUPLES[(2, 2, 0)] = 1

# Used only when fewer than 4 objects were detected (tuple does not sum to 4).
FALLBACK_LABEL = 1


def count_image(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (0, 0, 0)
    cls = boxes.cls.cpu().numpy().astype(int) # YOLO class ids: 0=cube, 1=sphere, 2=cylinder.
    conf = boxes.conf.cpu().numpy()
    order = conf.argsort()[::-1][:4]
    cls = cls[order]
    c = Counter(cls.tolist())
    return (c.get(1, 0), c.get(0, 0), c.get(2, 0)) #Count of (spheres, cubes, cylinders)


def label_for(triple):
    if triple in CLASS0_TUPLES:
        return 0
    if triple in CLASS1_TUPLES:
        return 1
    if triple in MISSING_TUPLES:
        return MISSING_TUPLES[triple]
    return FALLBACK_LABEL


def generate_predictions(data_dir):
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    # Some packagings nest the images one level deeper as test/test/*.png.
    nested = test_dir / "test"
    if nested.is_dir():
        has_imgs_here = any(p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                            for p in test_dir.iterdir() if p.is_file())
        if has_imgs_here:
            print(f"WARNING: both {test_dir} and {nested} contain images; "
                  f"using {nested}")
        test_dir = nested
    files = sorted(p for p in test_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"test images: {len(files)}")

    model = YOLO(str(WEIGHTS))

    rows = []
    batch = 64
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=480,
            verbose=False,
            conf=0.25,
        )
        for path, res in zip(chunk, results):
            triple = count_image(res)
            rows.append((path.name, label_for(triple)))
        print(f"  processed {min(i + batch, len(files))}/{len(files)}",
              flush=True)

    out_path = HERE / "submission.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        w.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = input("Enter path to data_dir: ").strip()
    generate_predictions(data_dir)
