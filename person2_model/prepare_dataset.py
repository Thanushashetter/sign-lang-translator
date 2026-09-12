import os
import cv2
import random
import shutil
import numpy as np
import sys

sys.path.append("person1_pose")

from extract_keypoints import extract_landmarks, create_landmarker

VIDEO_DIR = "data/videos/isl40"
OUTPUT_DIR = "data/processed"

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    landmarker = create_landmarker()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        landmarks = extract_landmarks(
            frame,
            landmarker
        )

        frames.append(landmarks)

    cap.release()
    landmarker.close()

    if len(frames) == 0:
        return None

    if len(frames) > SEQUENCE_LENGTH:

        indices = np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        frames = [frames[i] for i in indices]

    elif len(frames) < SEQUENCE_LENGTH:

        last = frames[-1]

        while len(frames) < SEQUENCE_LENGTH:
            frames.append(last)

    return np.array(
        frames,
        dtype=np.float32
    )


def main():

    for split in ["train", "val", "test"]:

        folder = os.path.join(
            OUTPUT_DIR,
            split
        )

        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)

    sign_folders = [
        f for f in os.listdir(VIDEO_DIR)
        if os.path.isdir(
            os.path.join(VIDEO_DIR, f)
        )
    ]

    sign_folders.sort()

    print("Total classes:", len(sign_folders))
    print()

    total = 0

    for sign in sign_folders:

        folder = os.path.join(
            VIDEO_DIR,
            sign
        )

        videos = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".mp4")
        ]

        random.shuffle(videos)

        n = len(videos)

        if n == 1:

            train_videos = videos
            val_videos = []
            test_videos = []

        elif n == 2:

            train_videos = videos[:1]
            val_videos = []
            test_videos = videos[1:]

        elif n == 3:

            train_videos = videos[:2]
            val_videos = []
            test_videos = videos[2:]

        else:

            train_end = max(
                1,
                int(n * TRAIN_RATIO)
            )

            val_count = max(
                1,
                int(n * VAL_RATIO)
            )

            if train_end + val_count >= n:
                val_count = 1

            train_videos = videos[:train_end]

            val_videos = videos[
                train_end:
                train_end + val_count
            ]

            test_videos = videos[
                train_end + val_count:
            ]

        splits = {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos
        }

        print(
            sign.upper(),
            "->",
            "Train:", len(train_videos),
            "Val:", len(val_videos),
            "Test:", len(test_videos)
        )

        for split, split_videos in splits.items():

            for i, filename in enumerate(split_videos):

                path = os.path.join(
                    folder,
                    filename
                )

                print(
                    "Processing:",
                    sign,
                    filename
                )

                try:

                    sequence = process_video(
                        path
                    )

                except Exception as e:

                    print(
                        "ERROR:",
                        filename
                    )

                    print(e)

                    continue

                if sequence is None:

                    print(
                        "Skipped:",
                        filename
                    )

                    continue

                class_name = sign.upper()

                output_name = (
                    class_name
                    + "__"
                    + str(i)
                    + ".npy"
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    split,
                    output_name
                )

                np.save(
                    output_path,
                    sequence
                )

                total += 1

    print()
    print("================================")
    print("ISL-40 DATASET PREPARATION DONE")
    print("================================")
    print("Total sequences:", total)


if __name__ == "__main__":
    main()