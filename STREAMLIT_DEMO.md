conda # Fish Detection Streamlit Demo

## Run

```powershell
conda activate fish
streamlit run app.py
```

Or run the included Windows helper:

```powershell
.\run_app.ps1
```

Then open:

```text
http://localhost:8501
```

The app uses this model by default:

```text
C:\Users\jinju\runs\detect\train-3\weights\best.pt
```

You can change the model path from the sidebar after retraining.

## Current Features

- Upload an image.
- Run YOLOv8 detection.
- Display bounding boxes, class names, and confidence scores.
- Show total fish count and species count.
- Show class-wise counts.
- Export detections to CSV.
- Optional rough length and adult/juvenile fields using a manual `cm per pixel` value.

## Notes

Keep `Centimetres per pixel` at `0` until the camera/depth calibration step is ready. Later, this value should come from the OAK-D depth pipeline instead of manual input.
