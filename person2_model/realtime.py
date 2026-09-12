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
import torch

from person1_pose.extract_keypoints import extract_landmarks
from person1_pose.preprocess import create_sequence

sys.path.append("person2_model")

from model import GlossModel


MODEL_FILE = "person2_model/models/gloss_model.pt"


checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

label_to_id = checkpoint["label_to_id"]
id_to_label = checkpoint["id_to_label"]


model = GlossModel(
    input_size=225,
    hidden_size=64,
    num_classes=len(label_to_id)
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


def predict(sequence):

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(sequence)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    return id_to_label[prediction]


cap = cv2.VideoCapture(0)

frames = []

prediction = "Waiting..."

print("Webcam started.")
print("Show a sign for 60 frames.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    frame_resized = cv2.resize(
        frame,
        (640, 480)
    )

    landmarks = extract_landmarks(
        frame_resized
    )

    frames.append(landmarks)

    if len(frames) == 60:

        try:

            sequence = create_sequence(
                frames
            )

            prediction = predict(
                sequence
            )

            print(
                "Predicted gloss:",
                prediction
            )

        except Exception as e:

            prediction = "Error"

            print(
                "Prediction error:",
                e
            )

        frames = []

    cv2.putText(
        display,
        "Prediction: " + prediction,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Frames: " + str(len(frames)) + "/60",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Sign Language Translator",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()