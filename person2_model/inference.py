import os
import numpy as np
import torch

from model import GlossModel

MODEL_PATH = "person2_model/models/gloss_model.pt"


def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    label_map = checkpoint["label_map"]

    reverse_map = {
        value: key
        for key, value in label_map.items()
    }

    model = GlossModel(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        len(label_map)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, reverse_map


def predict_gloss(sequence):

    model, reverse_map = load_model()

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (60, 225):
        raise ValueError(
            "Input must have shape (60, 225)"
        )

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(x)

        predicted = torch.argmax(
            output,
            dim=1
        ).item()

    return reverse_map[predicted]


def test_model():

    test_dir = "data/processed/test"

    files = [
        f for f in os.listdir(test_dir)
        if f.endswith(".npy")
    ]

    correct = 0
    total = 0

    for filename in sorted(files):

        sequence = np.load(
            os.path.join(
                test_dir,
                filename
            )
        )

        predicted = predict_gloss(sequence)

        actual = filename.rsplit(
            "__",
            1
        )[0]

        if predicted == actual:
            correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        total += 1

        print(
            actual,
            "->",
            predicted,
            result
        )

    print()
    print("==============================")
    print("TEST RESULTS")
    print("==============================")
    print(
        "Accuracy:",
        correct,
        "/",
        total
    )
    print(
        "Percentage:",
        round(100 * correct / total, 2),
        "%"
    )


if __name__ == "__main__":
    test_model()