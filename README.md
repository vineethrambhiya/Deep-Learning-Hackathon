# YOLO-CLEVR Submission

Binary classifier for the IITH Deep Learning 2026 Hackathon. A YOLOv8n detector (trained from scratch on CLEVR with scene-graph-derived boxes) counts the cubes / spheres / cylinders in each test image; the resulting tuple is mapped to a class label.

## Files

- `predict.py` — entry point with `generate_predictions(data_dir)`.
- `best.pt` — trained YOLOv8n weights.

## Expected `data_dir` layout

```
data_dir/
  train/
    0/...
    1/...
  test/
    *.png # or test/test/*.png — both are handled
```

Only `test/` is read at inference time.

## Setup

Create a fresh environment and install the dependencies:

```bash
conda create -n DL_hackathon_team_ASV python=3.10 -y
conda activate DL_hackathon_team_ASV
pip install ultralytics
```

## Running

All commands must be run from inside this folder (`team_ASV_submission/`), since `predict.py` resolves `best.pt` and writes `submission.csv` relative to its own location:

```bash
cd /path/to/team_ASV_submission
```

Either pass the data directory as a CLI argument:

```bash
python predict.py /path/to/data_dir
```

Or run it interactively:

```bash
python predict.py
/path/to/data_dir
```

The script writes `submission.csv` (columns `ID,Label`) into this folder.
That is the file we submit to Kaggle.
