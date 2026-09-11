import cv2
import numpy as np
from extract_keypoints import extract_landmarks
from preprocess import create_sequence

cap = cv2.VideoCapture(0)

frames = []

print("Collecting 60 frames...")
print("Perform your sign in front of the camera.")

while len(frames) < 60:

    ret, frame = cap.read()

    if not ret:
        break

    landmarks = extract_landmarks(frame)
    frames.append(landmarks)

    cv2.putText(
        frame,
        f"Frame: {len(frames)}/60",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("60 Frame Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(frames) == 60:

    sequence = create_sequence(frames)

    print("Sequence created successfully!")
    print("Shape:", sequence.shape)

    np.save("data/processed/test_sequence.npy", sequence)

    print("Saved to data/processed/test_sequence.npy")

else:
    print("Sequence incomplete.")