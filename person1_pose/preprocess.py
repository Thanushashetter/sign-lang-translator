import numpy as np


def normalize_landmarks(sequence):
    sequence = np.array(sequence, dtype=np.float32)

    mean = np.mean(sequence, axis=1, keepdims=True)
    std = np.std(sequence, axis=1, keepdims=True)

    std[std == 0] = 1

    return (sequence - mean) / std


def create_sequence(frames):
    if len(frames) != 60:
        raise ValueError("Sequence must contain exactly 60 frames")

    sequence = np.array(frames, dtype=np.float32)

    return normalize_landmarks(sequence)