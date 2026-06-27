"""
ASL WEBCAM INFERENCE — matched to mr.py v4
===========================================
Real-time ASL recognition using the model trained by mr.py v4.

Key fixes over old version:
  - Model head matches training (Linear 512 → ReLU → Linear N)
  - idx2class loaded FROM checkpoint (not rebuilt from excel)
  - num_classes auto-detected from checkpoint
  - Checkpoint path updated to checkpoints_v4

Controls:
    Q / ESC  - quit
    SPACE    - freeze/unfreeze prediction
    S        - save current frame as PNG
"""

import os
import sys
import cv2
import time
import urllib.request
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.video import r3d_18, R3D_18_Weights

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# =============================================================================
# CONFIG  — must match mr.py v4 training settings
# =============================================================================

NUM_FRAMES  = 8        # must match --num_frames used in training
IMG_SIZE    = 160      # must match --img_size used in training
PADDING     = 80
TILE_SIZE   = 32
CLIP_LIMIT  = 2.0
INFER_EVERY = 4        # run inference every N frames
TOP_K       = 5
CONF_THRESH = 0.10     # lowered slightly for 100-class model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# =============================================================================
# DOWNLOAD MEDIAPIPE MODEL
# =============================================================================

def download_hand_model():
    if os.path.isfile(MODEL_PATH):
        print(f"Hand model found: {MODEL_PATH}")
        return
    print("Downloading MediaPipe hand model (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# =============================================================================
# MODEL — must exactly match ASLModel in mr.py v4
# =============================================================================

class ASLModel(nn.Module):
    """
    Identical architecture to mr.py v4 ASLModel.
    Head: Dropout → Linear(512) → ReLU → Dropout → Linear(num_classes)
    """
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        backbone    = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

# =============================================================================
# PREPROCESSING  (identical to mr.py v4)
# =============================================================================

def clahe_luma(gray):
    h, w = gray.shape
    n_tiles_y = max(h // TILE_SIZE, 1)
    n_tiles_x = max(w // TILE_SIZE, 1)
    cdfs = [[None] * n_tiles_x for _ in range(n_tiles_y)]
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y1, x1 = ty * TILE_SIZE, tx * TILE_SIZE
            tile = gray[y1:y1+TILE_SIZE, x1:x1+TILE_SIZE]
            hist = np.zeros(256)
            for pi in range(tile.shape[0]):
                for pj in range(tile.shape[1]):
                    hist[tile[pi, pj]] += 1
            limit = CLIP_LIMIT * tile.size / 256
            excess = 0
            for k in range(256):
                if hist[k] > limit:
                    excess += hist[k] - limit
                    hist[k] = limit
            hist += int(excess // 256)
            cdf = np.zeros(256)
            cdf[0] = hist[0]
            for k in range(1, 256):
                cdf[k] = cdf[k-1] + hist[k]
            cdf_min = next((cdf[k] for k in range(256) if cdf[k] > 0), 0)
            cdf = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-5)
            cdfs[ty][tx] = np.clip(cdf, 0, 1)
    output = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            ty  = min(i // TILE_SIZE, n_tiles_y - 1)
            tx  = min(j // TILE_SIZE, n_tiles_x - 1)
            ty1 = min(ty + 1, n_tiles_y - 1)
            tx1 = min(tx + 1, n_tiles_x - 1)
            yr  = (i - ty * TILE_SIZE) / TILE_SIZE
            xr  = (j - tx * TILE_SIZE) / TILE_SIZE
            v   = gray[i, j]
            top    = cdfs[ty][tx][v]  * (1-xr) + cdfs[ty][tx1][v]  * xr
            bottom = cdfs[ty1][tx][v] * (1-xr) + cdfs[ty1][tx1][v] * xr
            output[i, j] = int((top*(1-yr) + bottom*yr) * 255)
    return output.astype(np.uint8)


def unsharp_mask(channel, amount=1.5, kernel_size=5):
    k  = kernel_size
    ax = np.arange(-(k//2), k//2+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / 2.0)
    kernel /= kernel.sum()
    pad    = k // 2
    padded = np.pad(channel.astype(np.float32), pad, mode="reflect")
    blurred = np.zeros_like(channel, dtype=np.float32)
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            blurred[i, j] = (padded[i:i+k, j:j+k] * kernel).sum()
    return np.clip(channel.astype(np.float32) + amount*(channel.astype(np.float32)-blurred), 0, 255).astype(np.uint8)


def preprocess_frame_color(frame_rgb):
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = unsharp_mask(clahe_luma(y))
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


def detect_hands(frame_rgb, detector):
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = detector.detect(mp_image)
    hands    = result.hand_landmarks

    # Fallback: try vertically flipped if no hands found
    if len(hands) < 1:
        flipped    = cv2.flip(frame_rgb, 0)
        mp_flipped = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped)
        result_flip = detector.detect(mp_flipped)
        if len(result_flip.hand_landmarks) > len(hands):
            flipped_hands = []
            for hand in result_flip.hand_landmarks:
                new_hand = [NormalizedLandmark(x=lm.x, y=1.0-lm.y, z=lm.z) for lm in hand]
                flipped_hands.append(new_hand)
            hands = flipped_hands

    return hands


def process_frame(frame_bgr, detector):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    hands   = detect_hands(frame_rgb, detector)
    n_found = len(hands)

    hand_bbox = None
    if hands:
        all_x, all_y = [], []
        for hand in hands:
            for lm in hand:
                all_x.append(lm.x * w)
                all_y.append(lm.y * h)
        x1 = max(int(min(all_x)) - PADDING, 0)
        y1 = max(int(min(all_y)) - PADDING, 0)
        x2 = min(int(max(all_x)) + PADDING, w)
        y2 = min(int(max(all_y)) + PADDING, h)
        hand_bbox = (x1, y1, x2, y2)
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_rgb

    resized       = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    enhanced      = preprocess_frame_color(resized)
    processed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    return processed_bgr, hand_bbox, n_found, hands, h, w


def frames_to_tensor(frames):
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = arr[:, :, :, ::-1].copy()          # BGR → RGB
    arr = (arr - MEAN) / STD
    t   = torch.from_numpy(arr).float().permute(3, 0, 1, 2)  # C T H W
    return t.unsqueeze(0)                                      # 1 C T H W

# =============================================================================
# DRAWING
# =============================================================================

def draw_hand_landmarks(frame, hands, h, w):
    for hand in hands:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 220, 100), 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 180, 80),    1,  cv2.LINE_AA)


def draw_bbox(frame, bbox, n_hands):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    color = (0, 255, 180) if n_hands == 2 else (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{n_hands} hand{'s' if n_hands!=1 else ''} detected",
                (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_predictions(frame, predictions, frozen):
    H, W   = frame.shape[:2]
    px, py = W - 340, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (px-10, py),
                  (W-5, py + 30 + len(predictions)*38 + 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "PREDICTION" + (" [FROZEN]" if frozen else ""),
                (px, py+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (lbl, conf) in enumerate(predictions):
        y = py + 40 + i*38
        cv2.rectangle(frame, (px, y), (px+200, y+18), (60, 60, 60), -1)
        cv2.rectangle(frame, (px, y), (px+int(200*conf), y+18),
                      (0, 220, 100) if i==0 else (80, 160, 220), -1)
        cv2.putText(frame, f"{lbl}  {conf*100:.1f}%", (px+4, y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255,255,255) if i==0 else (210,210,210), 1, cv2.LINE_AA)


def draw_status(frame, fps, buf_len, frozen, no_hands):
    H = frame.shape[0]
    parts = [f"FPS:{fps:.1f}", f"Buf:{buf_len}/{NUM_FRAMES}"]
    if frozen:   parts.append("FROZEN")
    if no_hands: parts.append("NO HANDS")
    cv2.putText(frame, "  |  ".join(parts), (10, H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:quit  SPACE:freeze  S:save", (10, H-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1, cv2.LINE_AA)

# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=r"D:\aslp\checkpoints_v4\best_model.pt")
    p.add_argument("--camera",     type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    download_hand_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Make sure training has finished and best_model.pt exists.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # idx2class and num_classes are saved inside the checkpoint by mr.py v4
    if "idx2class" not in ckpt:
        print("[ERROR] Checkpoint does not contain idx2class.")
        print("This checkpoint was likely saved by an older version of mr.py.")
        print("Please retrain with mr.py v4 or manually provide the class list.")
        sys.exit(1)

    idx2class   = ckpt["idx2class"]
    num_classes = ckpt["num_classes"]
    best_top1   = ckpt.get("val_top1", "?")
    best_epoch  = ckpt.get("epoch",    "?")

    print(f"Classes   : {num_classes}")
    print(f"Best val  : Top1 {best_top1:.2f}% (epoch {best_epoch})")
    print(f"Sample labels: {[idx2class[i] for i in range(min(5, num_classes))]}")

    # ── Build model with exact same architecture as mr.py v4 ─────────────────
    model = ASLModel(num_classes=num_classes, dropout=0.0)  # dropout=0 at inference
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("Model loaded and ready.")

    # ── MediaPipe HandLandmarker ──────────────────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.3,
    )
    detector = mp_vision.HandLandmarker.create_from_options(hand_opts)
    print("MediaPipe HandLandmarker ready.")

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera opened. Press Q or ESC to quit.")

    frame_buffer = collections.deque(maxlen=NUM_FRAMES)
    predictions  = []
    frames_since = 0
    frozen       = False
    fps_timer    = time.time()
    fps_val      = 0.0
    frame_count  = 0
    save_count   = 0

    while True:
        ret, raw_bgr = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        raw_bgr       = cv2.flip(raw_bgr, 1)   # mirror so it feels natural
        display_frame = raw_bgr.copy()

        proc_bgr, hand_bbox, n_hands, hands, h, w = process_frame(raw_bgr, detector)

        if not frozen:
            frame_buffer.append(proc_bgr)
            frames_since += 1

            if len(frame_buffer) == NUM_FRAMES and frames_since >= INFER_EVERY:
                frames_since = 0
                tensor = frames_to_tensor(list(frame_buffer)).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
                top_idx     = probs.argsort()[::-1][:TOP_K]
                predictions = [
                    (idx2class.get(int(i), f"Class_{i}"), float(probs[i]))
                    for i in top_idx
                    if float(probs[i]) >= CONF_THRESH
                ]

        draw_hand_landmarks(display_frame, hands, h, w)
        draw_bbox(display_frame, hand_bbox, n_hands)
        if predictions:
            draw_predictions(display_frame, predictions, frozen)
        draw_status(display_frame, fps_val, len(frame_buffer), frozen, n_hands == 0)

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_val     = frame_count / (now - fps_timer)
            fps_timer   = now
            frame_count = 0

        cv2.imshow("ASL Recognition — v4", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            frozen = not frozen
            print("Frozen" if frozen else "Unfrozen")
        elif key == ord('s'):
            fname = f"asl_capture_{save_count:04d}.png"
            cv2.imwrite(fname, display_frame)
            print(f"Saved: {fname}")
            save_count += 1

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

#pip install torch torchvision torchaudio
#py -3.11 -m venv mediapipe311
#pip install opencv-python mediapipe numpy torch torchvision torchaudio

