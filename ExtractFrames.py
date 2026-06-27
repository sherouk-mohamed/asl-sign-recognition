import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DATASET_CSV      = "D:\\aslp\\data\\cleaned_sign_dataset_cleaned.xlsx"
CLASS_STATS_XLSX = "D:\\aslp\\data\\asl_class_analysis_cleaned.xlsx"
VIDEO_FOLDER     = "D:\\aslp\\data\\batch_signs_video_v3_1"
OUTPUT_ROOT      = "D:\\aslp\\data\\final_videos"

FRAME_SIZE  = 224
PADDING     = 10
TILE_SIZE   = 32
CLIP_LIMIT  = 2.0


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna(subset=["Sign video filename", "Main entry gloss label"])
    df = df[df["Sign video filename"].str.endswith(".mp4", na=False)].copy()
    print(f"Dataset: {len(df)} rows | {df['Main entry gloss label'].nunique()} classes")
    return df


def load_class_stats(path: str) -> dict:
    """
    Returns a dict keyed by class name (uppercase):
      { "ABOUT": {"avg": 8.21, "small": 2, "large": 4}, ... }
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    stats = {}
    for _, row in df.iterrows():
        cls = str(row["Class"]).strip().upper()
        stats[cls] = {
            "avg":   float(row["Avg"]),
            "small": int(row["Small window size"]),
            "large": int(row["Large window size"]),
        }
    print(f"Class stats loaded for {len(stats)} classes")
    return stats


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────

def clahe_luma(gray: np.ndarray, tile_size: int = TILE_SIZE,
               clip_limit: float = CLIP_LIMIT) -> np.ndarray:
    h, w = gray.shape
    n_tiles_y = max(h // tile_size, 1)
    n_tiles_x = max(w // tile_size, 1)

    cdfs = [[None] * n_tiles_x for _ in range(n_tiles_y)]

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y1, x1 = ty * tile_size, tx * tile_size
            y2, x2 = y1 + tile_size, x1 + tile_size
            tile = gray[y1:y2, x1:x2]

            hist = np.zeros(256)
            for pi in range(tile.shape[0]):
                for pj in range(tile.shape[1]):
                    hist[tile[pi, pj]] += 1

            limit = clip_limit * tile.size / 256
            excess = 0
            for k in range(256):
                if hist[k] > limit:
                    excess += hist[k] - limit
                    hist[k] = limit
            hist += int(excess // 256)

            cdf = np.zeros(256)
            cdf[0] = hist[0]
            for k in range(1, 256):
                cdf[k] = cdf[k - 1] + hist[k]

            cdf_min = next((cdf[k] for k in range(256) if cdf[k] > 0), 0)
            cdf = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-5)
            cdf = np.clip(cdf, 0, 1)
            cdfs[ty][tx] = cdf

    output = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            ty = min(i // tile_size, n_tiles_y - 1)
            tx = min(j // tile_size, n_tiles_x - 1)
            ty1 = min(ty + 1, n_tiles_y - 1)
            tx1 = min(tx + 1, n_tiles_x - 1)

            y_ratio = (i - ty * tile_size) / tile_size
            x_ratio = (j - tx * tile_size) / tile_size
            val = gray[i, j]

            top    = cdfs[ty][tx][val]  * (1 - x_ratio) + cdfs[ty][tx1][val]  * x_ratio
            bottom = cdfs[ty1][tx][val] * (1 - x_ratio) + cdfs[ty1][tx1][val] * x_ratio
            output[i, j] = int((top * (1 - y_ratio) + bottom * y_ratio) * 255)

    return output.astype(np.uint8)


def unsharp_mask(channel: np.ndarray, amount: float = 1.5,
                 kernel_size: int = 5) -> np.ndarray:
    k = kernel_size
    sigma = 1.0
    ax = np.arange(-(k // 2), k // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    pad = k // 2
    padded = np.pad(channel.astype(np.float32), pad, mode="reflect")
    blurred = np.zeros_like(channel, dtype=np.float32)
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            blurred[i, j] = (padded[i:i+k, j:j+k] * kernel).sum()

    sharpened = channel.astype(np.float32) + amount * (channel.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_frame_color(frame_rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_enhanced = clahe_luma(y, tile_size=TILE_SIZE, clip_limit=CLIP_LIMIT)
    y_sharp    = unsharp_mask(y_enhanced, amount=1.5, kernel_size=5)
    merged = cv2.merge([y_sharp, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


# ──────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.3,
)


# ──────────────────────────────────────────────
# HAND DETECTION & ROI
# ──────────────────────────────────────────────

def detect_hands(frame_rgb: np.ndarray):
    result  = hands.process(frame_rgb)
    n_found = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0

    if n_found < 2:
        flipped     = cv2.flip(frame_rgb, 0)
        result_flip = hands.process(flipped)
        n_flip      = len(result_flip.multi_hand_landmarks) if result_flip.multi_hand_landmarks else 0

        if n_flip > n_found:
            for hl in result_flip.multi_hand_landmarks:
                for lm in hl.landmark:
                    lm.y = 1.0 - lm.y
            result = result_flip

    return result


def get_both_hands_roi(frame_rgb: np.ndarray, size: int = FRAME_SIZE,
                       padding: int = PADDING):
    result = detect_hands(frame_rgb)
    if not result.multi_hand_landmarks:
        return None

    h, w = frame_rgb.shape[:2]
    all_x, all_y = [], []

    for hand_landmarks in result.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(lm.x * w)
            all_y.append(lm.y * h)

    x_min = max(int(min(all_x)) - padding, 0)
    y_min = max(int(min(all_y)) - padding, 0)
    x_max = min(int(max(all_x)) + padding, w)
    y_max = min(int(max(all_y)) + padding, h)

    crop = frame_rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return preprocess_frame_color(crop_resized)


# ──────────────────────────────────────────────
# FRAME READING
# ──────────────────────────────────────────────

def read_all_frames(video_path: str) -> list:
    """Read every frame from a video as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────

def process_video(video_path: str, sign_label: str, video_stem: str,
                  class_stats: dict, output_root: str):

    all_frames = read_all_frames(video_path)
    n_frames   = len(all_frames)
    if n_frames == 0:
        return 0
    # Look up class stats (case-insensitive)
    cls_key = sign_label.strip().upper()
    if cls_key in class_stats:
        stats       = class_stats[cls_key]
        avg         = stats["avg"]
        window_size = stats["large"] if n_frames >= avg else stats["small"]
    else:
        # Fallback: no stats found — use 4 as default window
        window_size = 4
        print(f"  [WARN] No stats for class '{cls_key}', defaulting window_size=4")

    # Slide non-overlapping windows
    windows_saved = 0
    window_idx    = 0
    start         = 0

    while start + window_size <= n_frames:
        window_frames = all_frames[start : start + window_size]

        # Build output dir: output_root/sign_label/video_stem/window_NNN
        win_dir = os.path.join(output_root, sign_label, video_stem,
                               f"window_{window_idx:03d}")
        os.makedirs(win_dir, exist_ok=True)

        saved_in_window = 0
        for fi, frame in enumerate(window_frames):
            hand_roi = get_both_hands_roi(frame)
            if hand_roi is None:
                continue
            out_path = os.path.join(win_dir, f"frame_{fi:03d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(hand_roi, cv2.COLOR_RGB2BGR))
            saved_in_window += 1

        # Only count the window if at least one frame was saved
        if saved_in_window > 0:
            windows_saved += 1

        start      += window_size
        window_idx += 1

    return windows_saved


def run(dataset_path: str, class_stats_path: str,
        video_folder: str, output_root: str):

    df          = load_dataset(dataset_path)
    class_stats = load_class_stats(class_stats_path)

    total_videos  = 0
    total_windows = 0
    skipped       = 0

    for _, row in df.iterrows():
        video_name = str(row["Sign video filename"]).strip()
        sign_label = str(row["Main entry gloss label"]).strip()

        video_path = os.path.join(video_folder, video_name)
        if not os.path.isfile(video_path):
            skipped += 1
            continue

        # Use filename without extension as the per-video subfolder name
        video_stem = os.path.splitext(video_name)[0]

        n = process_video(video_path, sign_label, video_stem,
                          class_stats, output_root)
        total_windows += n
        total_videos  += 1

        if total_videos % 50 == 0:
            print(f"  Processed {total_videos} videos | {total_windows} windows saved")

    print(f"\nDone — {total_videos} videos | {total_windows} windows | "
          f"{skipped} missing videos skipped")
    print(f"Output: {output_root}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run(DATASET_CSV, CLASS_STATS_XLSX, VIDEO_FOLDER, OUTPUT_ROOT)