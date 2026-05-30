from collections import Counter
from pathlib import Path

base = Path(r"C:\Users\jinju\Downloads\archive (4)")

print("Checking path:", base)
print("Exists:", base.exists())

label_dirs = [
    base / "train" / "labels",
    base / "valid" / "labels",
    base / "test" / "labels"
]

img_count = 0
class_counts = Counter()

for label_dir in label_dirs:
    print("\nChecking:", label_dir)
    print("Exists:", label_dir.exists())

    if not label_dir.exists():
        print("❌ Folder missing!")
        continue

    files = list(label_dir.glob("*.txt"))
    print("Label files found:", len(files))

    for txt in files:
        img_count += 1
        with open(txt, "r") as f:
            for line in f:
                cls = int(line.split()[0])
                class_counts[cls] += 1

print("\n✅ FINAL RESULT")
print("Total labeled images:", img_count)
print("Class distribution:", class_counts)