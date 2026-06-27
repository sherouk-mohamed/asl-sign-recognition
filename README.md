# ASL Sign Language Recognition

Real-time American Sign Language recognition using a 3D ResNet-18 fine-tuned on ASL sign clips, with MediaPipe hand detection and a temporal sliding window approach.

---

## How It Works

1. **Frame extraction** — Videos are processed with MediaPipe to crop and enhance hand regions. A per-class sliding window splits each video into fixed-length clips.
2. **Window sizing** — Each ASL class gets its own optimal window size (small/large) based on the frame count distribution across its training videos.
3. **Training** — A Kinetics-400 pretrained `r3d_18` 3D ResNet is fine-tuned on the extracted clips using mixed-precision training, label smoothing, and cosine LR scheduling.
4. **Inference** — A live webcam feed is processed in real time: hands are detected, frames are buffered into windows, and the model predicts the sign every few frames.

---

## Project Structure

```
├── load.py            # Compute per-class sliding window sizes → Excel
├── ExtractFrames.py   # Extract & preprocess hand-crop windows from videos
├── train.py           # Fine-tune 3D ResNet-18 on extracted windows
├── webcam_asl.py      # Real-time webcam inference
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.11 recommended** (tested with MediaPipe on 3.11).

```bash
# Create and activate a virtual environment
py -3.11 -m venv mediapipe311
mediapipe311\Scripts\activate       # Windows
# source mediapipe311/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Compute window sizes

Run this once on your dataset CSV to generate per-class window sizes:

```bash
python load.py
```

Reads your video folder and outputs `asl_class_analysis.xlsx`.

### Step 2 — Extract training frames

```bash
python ExtractFrames.py
```

Edit the `CONFIG` section at the top of `ExtractFrames.py` to point to your dataset CSV, class stats Excel, video folder, and output directory.

Output structure:
```
final_videos/
  <CLASS>/
    <video_stem>/
      window_000/
        frame_000.jpg
        frame_001.jpg
        ...
```

### Step 3 — Train

```bash
python train.py \
  --data_root path/to/final_videos \
  --out_dir   path/to/checkpoints
```

Optional flags:
- `--epochs 60` — number of training epochs (default: 60)
- `--batch_size 8` — batch size (default: 8)
- `--resume path/to/checkpoint.pt` — resume from a saved checkpoint

Saves `best_model.pt` and `last_model.pt` to `--out_dir`, plus a `history.json` with loss/accuracy curves.

### Step 4 — Run webcam inference

```bash
python webcam_asl.py --checkpoint path/to/best_model.pt --camera 0
```

**Controls:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `SPACE` | Freeze / unfreeze prediction |
| `S` | Save current frame as PNG |

---

## Pretrained Model

Download `best_model.pt` and place it anywhere, then pass the path via `--checkpoint`:

> **[Download best_model.pt](#)** ← replace this with your Google Drive / Hugging Face link

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | `r3d_18` (Kinetics-400 pretrained) |
| Frames per clip | 8 |
| Frame size | 160 × 160 |
| Batch size | 8 |
| Max epochs | 60 |
| Learning rate | 1e-3 → 1e-6 (cosine) |
| Label smoothing | 0.1 |
| Early stopping patience | 10 |

---

## Requirements

See `requirements.txt`. Main dependencies: `torch`, `torchvision`, `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `openpyxl`.
