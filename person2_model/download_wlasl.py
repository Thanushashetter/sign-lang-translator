import json
import os
import subprocess


JSON_FILE = "data/raw/WLASL_v0.3.json"
VIDEO_DIR = "data/videos"


GLOSSES = [
    "BEFORE",
    "BOOK",
    "CANDY",
    "CHAIR",
    "CLOTHES",
    "COMPUTER",
    "COUSIN",
    "DRINK",
    "GO",
    "WHO"
]


TARGETS = {
    "train": 10,
    "val": 3,
    "test": 3
}


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


wanted = {
    gloss: item
    for item in data
    if (gloss := item["gloss"].upper()) in GLOSSES
}


total_attempted = 0
total_downloaded = 0
total_failed = 0


for gloss in GLOSSES:

    if gloss not in wanted:
        print("Missing gloss:", gloss)
        continue

    item = wanted[gloss]

    print()
    print("=" * 40)
    print(gloss)
    print("=" * 40)

    for split in ["train", "val", "test"]:

        output_dir = os.path.join(
            VIDEO_DIR,
            split
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        existing = [
            f for f in os.listdir(output_dir)
            if f.startswith(gloss + "_")
            and f.endswith(".mp4")
            and ".f" not in f
        ]

        count = len(existing)

        print(
            split,
            "existing:",
            count,
            "/",
            TARGETS[split]
        )

        if count >= TARGETS[split]:
            continue

        for instance in item["instances"]:

            if count >= TARGETS[split]:
                break

            if instance["split"] != split:
                continue

            video_id = instance["video_id"]
            url = instance["url"]

            filename = f"{gloss}_{video_id}.mp4"

            output_file = os.path.join(
                output_dir,
                filename
            )

            if os.path.exists(output_file):
                count += 1
                continue

            total_attempted += 1

            print(
                "Trying:",
                split,
                video_id
            )

            command = [
                "py",
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--no-part",
                "--retries",
                "2",
                "--fragment-retries",
                "2",
                "--quiet",
                "--no-warnings",
                "-o",
                output_file,
                url
            ]

            result = subprocess.run(
                command
            )

            if (
                result.returncode == 0
                and os.path.exists(output_file)
                and os.path.getsize(output_file) > 10000
            ):

                count += 1
                total_downloaded += 1

                print(
                    "Downloaded:",
                    filename
                )

            else:

                total_failed += 1

                if os.path.exists(output_file):
                    os.remove(output_file)

                print(
                    "Failed:",
                    video_id
                )


print()
print("=" * 40)
print("DOWNLOAD COMPLETE")
print("=" * 40)
print("Attempted :", total_attempted)
print("Downloaded:", total_downloaded)
print("Failed    :", total_failed)
print("=" * 40)