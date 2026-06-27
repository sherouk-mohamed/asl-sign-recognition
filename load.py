
##this is how i got the size of the windoes
import os
import cv2
import pandas as pd

# ============================================
# LOAD CSV
# ============================================

df = pd.read_csv(
    r'D:\aslp\data\asllrp_sentence_signs_2025_06_28.csv'
)

# ============================================
# SETTINGS
# ============================================

LABEL_COLUMN = 'Main entry gloss label'
VIDEO_COLUMN = 'Sign video filename'

VIDEO_FOLDER = r'D:\aslp\data\batch_signs_video_v3_1'

EXCEL_FILE = 'asl_class_analysis.xlsx'

# ============================================
# FUNCTION: COUNT FRAMES IN VIDEO
# ============================================

def count_video_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    return total_frames

# ============================================
# FUNCTION: COMPUTE WINDOW SIZES
# ============================================

def largest_pow2_sliding(limit):
    w = 1
    while w * 2 <= limit / 1.5:
        w *= 2
    return max(1, w)

def get_windows(min_frames, avg_frames, max_frames, total_videos):

    small_limit = min_frames
    large_limit = max_frames if total_videos <= 3 else avg_frames

    small_window = largest_pow2_sliding(small_limit)
    large_window = largest_pow2_sliding(large_limit)

    # make sure they are different
    if large_window <= small_window:
        large_window = small_window * 2

    return small_window, large_window

# ============================================
# PROCESS CLASSES
# ============================================

results = []

classes = df[LABEL_COLUMN].unique()

print(f"Total number of classes: {len(classes)}")

for class_name in classes:

    print(f"\nProcessing class: {class_name}")

    class_df = df[
        df[LABEL_COLUMN] == class_name
    ]

    frame_counts = []

    # ========================================
    # LOOP THROUGH VIDEOS
    # ========================================

    for _, row in class_df.iterrows():

        video_filename = row[VIDEO_COLUMN]

        video_path = os.path.join(
            VIDEO_FOLDER,
            video_filename
        )

        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue

        frame_counts.append(
            count_video_frames(video_path)
        )

    # ========================================
    # FILTER: drop videos below avg / 2
    # ========================================

    if frame_counts:

        raw_avg   = sum(frame_counts) / len(frame_counts)
        threshold = raw_avg / 2

        frame_counts = [
            f for f in frame_counts
            if f >= threshold
        ]

    # ========================================
    # COMPUTE STATS
    # ========================================

    total_videos = len(frame_counts)

    if total_videos > 0:

        min_frames    = min(frame_counts)
        max_frames    = max(frame_counts)
        avg_frames    = round(
            sum(frame_counts) / total_videos, 2
        )

        small_window, large_window = get_windows(
            min_frames,
            avg_frames,
            max_frames,
            total_videos
        )

    else:

        min_frames    = 0
        max_frames    = 0
        avg_frames    = 0
        small_window  = 0
        large_window  = 0

    # ========================================
    # SAVE RESULTS
    # ========================================

    results.append({
        'Class':              class_name,
        'Min':                min_frames,
        'Avg':                avg_frames,
        'Max':                max_frames,
        'Vids':               total_videos,
        'Small window size':  small_window,
        'Large window size':  large_window,
    })

# ============================================
# CREATE DATAFRAME
# ============================================

results_df = pd.DataFrame(results)

# ============================================
# SORT BY CLASS NAME
# ============================================

results_df = results_df.sort_values(
    by='Class'
).reset_index(drop=True)

# ============================================
# SAVE TO EXCEL
# ============================================

results_df.to_excel(
    EXCEL_FILE,
    index=False
)

print(f"\nExcel file saved as: {EXCEL_FILE}")
print("\nDone.")
print(results_df.head())