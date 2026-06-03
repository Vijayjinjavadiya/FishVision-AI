# pyrefly: ignore [missing-import]
import cv2
from src.veto_gate import VetoGate
from src.biometrics import MAX_LENGTH_CM

gate = VetoGate()

# Test phone-screen images
imgs = [
    ("fish_0034_GoldFish",  r"c:\fish_detection\logs\captures\fish_0034_GoldFish_2026-06-03_11-18-07.jpg"),
    ("fish_0043_ClownFish", r"c:\fish_detection\logs\captures\fish_0043_ClownFish_2026-06-03_11-18-26.jpg"),
]
print("Phone-screen images (expect cell phone / person veto):")
for n, p in imgs:
    frame = cv2.imread(p)
    veto_boxes = gate.detect_veto_objects(frame)
    classes = [b[4] for b in veto_boxes]
    print(f"  {n}: veto objects = {classes}")

print()
print("Size filter validation:")
pairs = [
    ("ClownFish", 154.3), ("Gourami", 108.8), ("BlueTang", 99.6),
    ("PlatyFish", 9.0), ("ClownFish", 9.1),
]
for s, l in pairs:
    mx = MAX_LENGTH_CM[s]
    status = "BLOCKED" if l > mx else "OK"
    print(f"  {s} {l}cm vs max {mx}cm -> {status}")
