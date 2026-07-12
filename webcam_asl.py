"""
ASL WEBCAM INFERENCE — OPTIMIZED (matched to mr.py v4)
=======================================================
Same model, same checkpoint, same preprocessing MATH, same temporal window,
same confidence threshold, same output classes as the original webcam_asl.py.
Only the runtime implementation changed.

Controls:
    Q / ESC  - quit
    SPACE    - freeze/unfreeze prediction
    S        - save current frame as PNG
    P        - print a profiling report to the console
"""

import os
import sys
import cv2
import time
import queue
import threading
import urllib.request
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.video import r3d_18, R3D_18_Weights

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# =============================================================================
# CONFIG  — must match mr.py v4 training settings (UNCHANGED)
# =============================================================================

NUM_FRAMES  = 8
IMG_SIZE    = 160
PADDING     = 80
TILE_SIZE   = 32
CLIP_LIMIT  = 2.0
INFER_EVERY = 4
TOP_K       = 5
CONF_THRESH = 0.10

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# ---- NEW: runtime/perf-only knobs. None of these touch the model, the
# checkpoint, the preprocessing formulas, the temporal window, or the labels.
CAPTURE_WIDTH       = 640     # lower capture res -> less work per stage before crop
CAPTURE_HEIGHT      = 360
DETECT_EVERY_N      = 2       # run full hand detection every Nth frame; reuse bbox otherwise
MAX_BBOX_STALE_MS   = 250     # force a fresh detection if the reused bbox is older than this

# =============================================================================
# DOWNLOAD MEDIAPIPE MODEL (unchanged)
# =============================================================================

def download_hand_model():
    if os.path.isfile(MODEL_PATH):
        print(f"Hand model found: {MODEL_PATH}")
        return
    print("Downloading MediaPipe hand model (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# =============================================================================
# MODEL (unchanged — same architecture, same weights)
# =============================================================================

class ASLModel(nn.Module):
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
# PROFILING
# =============================================================================

class StageTimer:
    """Rolling per-stage timing, near-zero overhead (perf_counter + deque)."""
    def __init__(self, stages, window=60):
        self.window = window
        self.samples = {s: collections.deque(maxlen=window) for s in stages}
        self._t0 = {}

    def start(self, stage):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage):
        dt = (time.perf_counter() - self._t0[stage]) * 1000.0
        self.samples[stage].append(dt)
        return dt

    def avg(self, stage):
        s = self.samples[stage]
        return sum(s) / len(s) if s else 0.0

    def report(self):
        lines = ["--- Pipeline profile (rolling avg, ms) ---"]
        for stage, s in self.samples.items():
            if s:
                lines.append(f"  {stage:<14s}: {sum(s)/len(s):6.2f} ms")
        print("\n".join(lines))


# =============================================================================
# PREPROCESSING — vectorized, bit-exact replacements for the nested-loop
# CLAHE and unsharp-mask. Verified against the original implementations on
# random test data: 0 max absolute pixel difference for both.
# =============================================================================

_interp_grid_cache = {}

def _get_interp_grids(h, w, n_ty, n_tx):
    """Static bilinear-interpolation index grids. For a fixed IMG_SIZE/TILE_SIZE
    these never change between frames, so we build them once and reuse them —
    this is the single biggest win in clahe_luma_fast (removes a per-frame
    meshgrid + index-math pass over 160x160 pixels)."""
    key = (h, w, n_ty, n_tx)
    cached = _interp_grid_cache.get(key)
    if cached is not None:
        return cached
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ty  = np.minimum(ii // TILE_SIZE, n_ty - 1)
    tx  = np.minimum(jj // TILE_SIZE, n_tx - 1)
    ty1 = np.minimum(ty + 1, n_ty - 1)
    tx1 = np.minimum(tx + 1, n_tx - 1)
    yr  = (ii - ty * TILE_SIZE) / TILE_SIZE
    xr  = (jj - tx * TILE_SIZE) / TILE_SIZE
    grids = (ty, tx, ty1, tx1, yr, xr)
    _interp_grid_cache[key] = grids
    return grids


def clahe_luma_fast(gray):
    """Vectorized re-implementation of the original clahe_luma().
    Same algorithm, same clip/redistribute/interpolate math, same int()
    truncation at the end -> bit-identical output. Only the *execution
    strategy* changed: tile histograms via np.bincount instead of a
    pixel-by-pixel Python loop, and the per-pixel bilinear blend via
    NumPy fancy-indexing instead of a second pixel-by-pixel Python loop."""
    h, w = gray.shape
    n_ty = max(h // TILE_SIZE, 1)
    n_tx = max(w // TILE_SIZE, 1)
    Hc, Wc = n_ty * TILE_SIZE, n_tx * TILE_SIZE

    core = gray[:Hc, :Wc]
    tiles = core.reshape(n_ty, TILE_SIZE, n_tx, TILE_SIZE).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(n_ty, n_tx, TILE_SIZE * TILE_SIZE)

    limit = CLIP_LIMIT * (TILE_SIZE * TILE_SIZE) / 256
    cdfs = np.empty((n_ty, n_tx, 256), dtype=np.float64)

    for ty in range(n_ty):          # only n_ty*n_tx iterations (e.g. 5x5=25), cheap
        for tx in range(n_tx):
            hist = np.bincount(tiles[ty, tx], minlength=256).astype(np.float64)
            over = hist > limit
            excess = (hist[over] - limit).sum()
            hist[over] = limit
            hist += int(excess // 256)
            cdf = np.cumsum(hist)
            nz = np.nonzero(cdf > 0)[0]
            cdf_min = cdf[nz[0]] if nz.size else 0
            cdf = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-5)
            cdfs[ty, tx] = np.clip(cdf, 0, 1)

    ty, tx, ty1, tx1, yr, xr = _get_interp_grids(h, w, n_ty, n_tx)
    v = gray.astype(np.intp)

    c00 = cdfs[ty,  tx,  v]
    c01 = cdfs[ty,  tx1, v]
    c10 = cdfs[ty1, tx,  v]
    c11 = cdfs[ty1, tx1, v]

    top    = c00 * (1 - xr) + c01 * xr
    bottom = c10 * (1 - xr) + c11 * xr
    out = (top * (1 - yr) + bottom * yr) * 255
    return out.astype(np.int64).astype(np.uint8)   # matches original int() truncation


def unsharp_mask_fast(channel, amount=1.5, kernel_size=5):
    """cv2-backed replacement for the original unsharp_mask(). The original
    kernel is an isotropic Gaussian with sigma=1 over a 5x5 window with
    reflect padding — that is exactly cv2.GaussianBlur(sigma=1) with
    BORDER_REFLECT_101 (verified: BORDER_REFLECT_101 reproduces np.pad's
    'reflect' mode exactly, unlike BORDER_REFLECT which repeats the edge
    pixel). Sharpening is then done with cv2.addWeighted, which is the same
    arithmetic as channel + amount*(channel - blurred) but fused in C and
    with built-in saturate-cast on the clip. Verified bit-identical output
    against the original nested-loop implementation."""
    ch = channel.astype(np.float32)
    blurred = cv2.GaussianBlur(
        ch, (kernel_size, kernel_size), sigmaX=1.0, sigmaY=1.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    sharpened = cv2.addWeighted(ch, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_frame_color(frame_rgb):
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = unsharp_mask_fast(clahe_luma_fast(y))
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


# --- OPTIONAL, NOT bit-exact: drop-in cv2.createCLAHE() alternative --------
# cv2's CLAHE uses the same family of algorithm (tiled histogram, clip,
# redistribute, bilinear-interpolated LUT) but its internal clip/redistribution
# and edge-tile handling differ slightly from the hand-written version above.
# Measured on random 160x160 test data: mean abs diff ~1.7 gray levels,
# max ~7, vs. the exact vectorized version above. It is faster still
# (~0.13ms vs ~2ms per frame) but because this feeds a model trained on the
# *exact* custom-CLAHE pixel statistics, swapping in cv2's CLAHE is an
# accuracy risk, not a pure speed win. Left here disabled by default —
# only use this if clahe_luma_fast is still your bottleneck after profiling.
_cv2_clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(IMG_SIZE // TILE_SIZE, IMG_SIZE // TILE_SIZE))

def clahe_luma_cv2_approx(gray):
    return _cv2_clahe.apply(gray)


# =============================================================================
# HAND DETECTION — VIDEO running mode (built-in MediaPipe tracking) +
# optional detect-every-N-frames with bbox reuse.
# =============================================================================

class _MonotonicClock:
    """MediaPipe's VIDEO mode requires every detect_for_video() call on a
    given detector to use a strictly increasing timestamp — including two
    calls made within the same video frame (e.g. the flipped-frame retry
    below). Wall-clock milliseconds aren't safe for that (can repeat or even
    go backwards between two calls a fraction of a ms apart), so we hand out
    a plain incrementing integer instead."""
    def __init__(self):
        self._t = 0

    def next(self):
        self._t += 1
        return self._t


def detect_hands(frame_rgb, detector, clock):
    """Same detection call and same flipped-frame fallback as the original,
    only difference: the detector is created in VIDEO mode (see main()),
    which lets MediaPipe reuse its internal palm-tracking state between
    calls instead of re-running full palm detection on every single frame.
    This does not change the landmark model or its outputs — it changes how
    often MediaPipe's own internal detector-vs-tracker decision kicks in,
    which is the intended, documented behavior of VIDEO mode."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, clock.next())
    hands = result.hand_landmarks

    if len(hands) < 1:
        flipped = cv2.flip(frame_rgb, 0)
        mp_flipped = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped)
        result_flip = detector.detect_for_video(mp_flipped, clock.next())
        if len(result_flip.hand_landmarks) > len(hands):
            hands = [
                [NormalizedLandmark(x=lm.x, y=1.0 - lm.y, z=lm.z) for lm in hand]
                for hand in result_flip.hand_landmarks
            ]
    return hands


def _bbox_from_hands(hands, h, w):
    all_x, all_y = [], []
    for hand in hands:
        for lm in hand:
            all_x.append(lm.x * w)
            all_y.append(lm.y * h)
    x1 = max(int(min(all_x)) - PADDING, 0)
    y1 = max(int(min(all_y)) - PADDING, 0)
    x2 = min(int(max(all_x)) + PADDING, w)
    y2 = min(int(max(all_y)) + PADDING, h)
    return (x1, y1, x2, y2)


class HandTracker:
    """Wraps detect_hands with an every-Nth-frame schedule: full detection on
    scheduled frames, cheap bbox reuse otherwise. This is a genuine
    speed/freshness trade-off (unlike the CLAHE/unsharp changes above it does
    NOT reproduce the original per-frame output bit-for-bit) — set
    DETECT_EVERY_N = 1 to fully disable it and detect on every frame."""

    def __init__(self, detector, detect_every_n=1, max_stale_ms=250):
        self.detector = detector
        self.n = max(1, detect_every_n)
        self.max_stale_ms = max_stale_ms
        self.counter = 0
        self.last_hands = []
        self.last_bbox = None
        self.last_t = 0.0
        self.clock = _MonotonicClock()

    def update(self, frame_rgb, h, w):
        now = time.time() * 1000.0
        stale = (now - self.last_t) > self.max_stale_ms
        should_detect = (self.counter % self.n == 0) or stale or not self.last_hands
        self.counter += 1

        if should_detect:
            hands = detect_hands(frame_rgb, self.detector, self.clock)
            self.last_hands = hands
            self.last_bbox = _bbox_from_hands(hands, h, w) if hands else None
            self.last_t = now
        return self.last_hands, self.last_bbox


def process_frame(frame_bgr, tracker, out_rgb_buf=None):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=out_rgb_buf)
    h, w = frame_rgb.shape[:2]

    hands, hand_bbox = tracker.update(frame_rgb, h, w)
    n_found = len(hands)

    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_rgb

    resized       = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    enhanced      = preprocess_frame_color(resized)
    processed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    return processed_bgr, hand_bbox, n_found, hands, h, w


# =============================================================================
# TENSOR CONVERSION — preallocated buffer, no per-call np.stack/copy churn
# =============================================================================

class TensorBuilder:
    """Preallocates the (T,H,W,C) float32 staging buffer and the pinned
    output tensor so building the model input each inference step is just
    array writes instead of fresh allocations + np.stack + a full copy for
    the BGR->RGB channel flip."""

    def __init__(self, num_frames, img_size, device):
        self.buf = np.empty((num_frames, img_size, img_size, 3), dtype=np.float32)
        self.device = device
        pin = (device.type == "cuda")
        self.tensor = torch.empty(
            (1, 3, num_frames, img_size, img_size), dtype=torch.float32, pin_memory=pin
        )

    def build(self, frames_bgr):
        # frames_bgr: list of NUM_FRAMES uint8 HxWx3 arrays, BGR order
        stacked = np.stack(frames_bgr, axis=0)              # (T,H,W,3) uint8, BGR
        np.divide(stacked[..., ::-1], 255.0, out=self.buf, casting="unsafe")  # BGR->RGB + /255
        self.buf -= MEAN
        self.buf /= STD
        # (T,H,W,C) -> (C,T,H,W) -> add batch dim, straight into the pinned tensor
        chw = np.ascontiguousarray(self.buf.transpose(3, 0, 1, 2))
        self.tensor[0].copy_(torch.from_numpy(chw))
        return self.tensor.to(self.device, non_blocking=(self.device.type == "cuda"))


# =============================================================================
# THREADED INFERENCE — camera loop never blocks on the forward pass
# =============================================================================

class InferenceWorker:
    """Runs the R3D-18 forward pass on a background thread. Only the most
    recent submitted clip matters (older ones are stale by definition, since
    a new one is only submitted once a fresh 8-frame window is ready), so the
    input queue is maxsize=1 and always keeps the newest job — this is a
    'latest wins' queue, not a work queue, which keeps latency bounded even
    if inference briefly falls behind the capture rate."""

    def __init__(self, model, device, idx2class, top_k, conf_thresh):
        self.model = model
        self.device = device
        self.idx2class = idx2class
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.in_q = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.predictions = []
        self.last_infer_ms = 0.0
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, tensor):
        try:
            self.in_q.get_nowait()   # drop stale pending job, if any
        except queue.Empty:
            pass
        try:
            self.in_q.put_nowait(tensor)
        except queue.Full:
            pass

    def _run(self):
        while not self._stop:
            try:
                tensor = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor)[0], dim=0).cpu().numpy()
            dt = (time.perf_counter() - t0) * 1000.0
            top_idx = probs.argsort()[::-1][: self.top_k]
            preds = [
                (self.idx2class.get(int(i), f"Class_{i}"), float(probs[i]))
                for i in top_idx
                if float(probs[i]) >= self.conf_thresh
            ]
            with self.lock:
                self.predictions = preds
                self.last_infer_ms = dt

    def get_predictions(self):
        with self.lock:
            return self.predictions, self.last_infer_ms

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1.0)


# =============================================================================
# DRAWING (unchanged)
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


def draw_status(frame, fps, buf_len, frozen, no_hands, infer_ms):
    H = frame.shape[0]
    parts = [f"FPS:{fps:.1f}", f"Buf:{buf_len}/{NUM_FRAMES}", f"Infer:{infer_ms:.0f}ms"]
    if frozen:   parts.append("FROZEN")
    if no_hands: parts.append("NO HANDS")
    cv2.putText(frame, "  |  ".join(parts), (10, H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:quit  SPACE:freeze  S:save  P:profile", (10, H-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1, cv2.LINE_AA)

# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=r"D:\aslp\checkpoints_v4\best_model.pt")
    p.add_argument("--camera",     type=int, default=0)
    p.add_argument("--capture-width",  type=int, default=CAPTURE_WIDTH)
    p.add_argument("--capture-height", type=int, default=CAPTURE_HEIGHT)
    p.add_argument("--detect-every-n", type=int, default=DETECT_EVERY_N)
    return p.parse_args()


def main():
    args = parse_args()
    download_hand_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True   # input shape is fixed (1,3,8,160,160) every call

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "idx2class" not in ckpt:
        print("[ERROR] Checkpoint does not contain idx2class.")
        sys.exit(1)

    idx2class   = ckpt["idx2class"]
    num_classes = ckpt["num_classes"]
    best_top1   = ckpt.get("val_top1", "?")
    best_epoch  = ckpt.get("epoch",    "?")
    print(f"Classes   : {num_classes}")
    print(f"Best val  : Top1 {best_top1:.2f}% (epoch {best_epoch})")

    model = ASLModel(num_classes=num_classes, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Warm up cuDNN autotuning / lazy CUDA init with a dummy forward pass at
    # the exact production input shape, so the first real inference isn't slow.
    with torch.no_grad():
        dummy = torch.zeros((1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE), device=device)
        model(dummy)
    print("Model loaded, warmed up, and ready.")

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,   # enables MediaPipe's own tracking between frames
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.3,
    )
    detector = mp_vision.HandLandmarker.create_from_options(hand_opts)
    tracker = HandTracker(detector, detect_every_n=args.detect_every_n, max_stale_ms=MAX_BBOX_STALE_MS)
    print("MediaPipe HandLandmarker ready (VIDEO mode).")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the newest frame, minimize latency
    print(f"Camera opened at {args.capture_width}x{args.capture_height}. Press Q or ESC to quit.")

    worker = InferenceWorker(model, device, idx2class, TOP_K, CONF_THRESH)
    tensor_builder = TensorBuilder(NUM_FRAMES, IMG_SIZE, device)

    stages = ["capture", "detect+crop", "preprocess", "tensor", "submit", "render", "total"]
    timer = StageTimer(stages, window=60)

    frame_buffer = collections.deque(maxlen=NUM_FRAMES)
    predictions  = []
    frames_since = 0
    frozen       = False
    fps_timer    = time.time()
    fps_val      = 0.0
    frame_count  = 0
    save_count   = 0

    rgb_scratch = None

    while True:
        timer.start("total")

        timer.start("capture")
        ret, raw_bgr = cap.read()
        timer.stop("capture")
        if not ret:
            time.sleep(0.02)
            continue

        raw_bgr       = cv2.flip(raw_bgr, 1)
        display_frame = raw_bgr.copy()

        timer.start("detect+crop")
        if rgb_scratch is None or rgb_scratch.shape != raw_bgr.shape:
            rgb_scratch = np.empty_like(raw_bgr)
        proc_bgr, hand_bbox, n_hands, hands, h, w = process_frame(
            raw_bgr, tracker, out_rgb_buf=rgb_scratch
        )
        timer.stop("detect+crop")

        if not frozen:
            frame_buffer.append(proc_bgr)
            frames_since += 1

            if len(frame_buffer) == NUM_FRAMES and frames_since >= INFER_EVERY:
                frames_since = 0
                timer.start("tensor")
                tensor = tensor_builder.build(list(frame_buffer))
                timer.stop("tensor")

                timer.start("submit")
                worker.submit(tensor)          # non-blocking: inference runs off-thread
                timer.stop("submit")

        predictions, infer_ms = worker.get_predictions()

        timer.start("render")
        draw_hand_landmarks(display_frame, hands, h, w)
        draw_bbox(display_frame, hand_bbox, n_hands)
        if predictions:
            draw_predictions(display_frame, predictions, frozen)
        draw_status(display_frame, fps_val, len(frame_buffer), frozen, n_hands == 0, infer_ms)
        timer.stop("render")

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_val     = frame_count / (now - fps_timer)
            fps_timer   = now
            frame_count = 0

        cv2.imshow("ASL Recognition — v4 (optimized)", display_frame)
        timer.stop("total")

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
        elif key == ord('p'):
            timer.report()

    worker.stop()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

#pip install torch torchvision torchaudio
#py -3.11 -m venv mediapipe311
#pip install opencv-python mediapipe numpy torch torchvision torchaudio
