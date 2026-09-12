import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2
import numpy as np

from person1_pose.extract_keypoints import extract_landmarks


video = "data/videos/test/BEFORE_05727.mp4"

cap = cv2.VideoCapture(video)

total = 0
hands = 0
nonzero = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    landmarks = extract_landmarks(
        frame
    )

    total += 1

    if np.count_nonzero(
        landmarks[:126]
    ) > 0:

        hands += 1

    if np.count_nonzero(
        landmarks
    ) > 0:

        nonzero += 1


cap.release()


print()
print("Total frames:", total)
print("Frames with hand data:", hands)
print("Frames with any landmark data:", nonzero)