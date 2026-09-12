import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import GlossModel


TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
MODEL_DIR = "person2_model/models"

INPUT_SIZE = 225
HIDDEN_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 75
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


def get_label(filename):
    return filename.rsplit("__", 1)[0]


def load_data(folder, label_map=None):

    X = []
    y = []

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    if label_map is None:

        labels = sorted(
            set(get_label(f) for f in files)
        )

        label_map = {
            label: i
            for i, label in enumerate(labels)
        }

    for filename in files:

        sequence = np.load(
            os.path.join(folder, filename)
        )

        label = get_label(filename)

        X.append(sequence)
        y.append(label_map[label])

    X = torch.tensor(
        np.array(X),
        dtype=torch.float32
    )

    y = torch.tensor(
        np.array(y),
        dtype=torch.long
    )

    return X, y, label_map


def main():

    X_train, y_train, label_map = load_data(
        TRAIN_DIR
    )

    X_val, y_val, _ = load_data(
        VAL_DIR,
        label_map
    )

    print("Classes:", len(label_map))
    print(sorted(label_map.keys()))

    print("Train samples:", len(X_train))
    print("Validation samples:", len(X_val))

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    model = GlossModel(
        INPUT_SIZE,
        HIDDEN_SIZE,
        len(label_map)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val = 0

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total = 0
        loss_sum = 0

        for X, y in train_loader:

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item()

            predicted = torch.argmax(
                output,
                dim=1
            )

            total += y.size(0)

            correct += (
                predicted == y
            ).sum().item()

        train_acc = 100 * correct / total

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X, y in val_loader:

                output = model(X)

                predicted = torch.argmax(
                    output,
                    dim=1
                )

                total += y.size(0)

                correct += (
                    predicted == y
                ).sum().item()

        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {loss_sum/len(train_loader):.4f} "
            f"Train: {train_acc:.2f}% "
            f"Val: {val_acc:.2f}%"
        )

        if val_acc > best_val:

            best_val = val_acc

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_map": label_map,
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE
                },
                os.path.join(
                    MODEL_DIR,
                    "gloss_model.pt"
                )
            )

            print("Best model saved!")

    print()
    print("Training complete!")
    print(
        "Best validation accuracy:",
        best_val
    )


if __name__ == "__main__":
    main()