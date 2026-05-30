# Architecture Notes

This document covers design decisions and the planned evolution of the Fish Detection System. It's meant to be a living reference — not a formal spec.

---

## Current architecture (Phase 1)

```
┌─────────────────────────────────────────────────────┐
│                     app.py                          │
│           (Streamlit orchestration layer)           │
└────────────┬──────────────────────────┬─────────────┘
             │                          │
             ▼                          ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │  src/detector   │      │  src/visualization   │
    │  • load_model() │      │  • render_metrics()  │
    │  • run_infer()  │      │  • render_image()    │
    └────────┬────────┘      │  • render_table()    │
             │               └──────────────────────┘
             ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │  src/predictor  │      │    src/utils          │
    │  • detections   │      │  • path resolution   │
    │    _to_df()     │      │  • dataset checks    │
    │  • summary()    │      └──────────────────────┘
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  models/best.pt │
    │  YOLOv8n weights│
    └─────────────────┘
```

### Why split into modules?

The original `app.py` mixed model loading, inference, post-processing, and UI rendering in a single 189-line file. That works fine for a quick demo but becomes a problem the moment you want to:

- Unit test the detection logic without a browser
- Swap the UI framework (Gradio, FastAPI, etc.)
- Add a CLI training script that reuses the same predictor
- Profile which part of the pipeline is slow

Each module in `src/` has a single responsibility:

| Module | Owns |
|--------|------|
| `detector.py` | Everything ultralytics-specific |
| `predictor.py` | YOLO results → tidy DataFrames |
| `utils.py` | Paths, filesystem checks |
| `visualization.py` | Everything Streamlit-specific |

---

## Planned phases

### Phase 2 — Length estimation

The `cm_per_pixel` input in the sidebar is already wired up. The missing piece is the source of that value.

- **Short term:** Let the user measure a reference object in the frame and compute the scale manually.
- **Long term:** Read depth data from the OAK-D Pro and compute `cm_per_pixel` automatically per-frame.

The predictor already handles the math (`width_px * cm_per_pixel`). Phase 2 is purely about getting a reliable `cm_per_pixel` value.

### Phase 3 — Weight estimation

Fish weight correlates strongly with length. A simple allometric formula (`W = a * L^b`) per species is a reasonable first step.

Species-specific `a` and `b` constants will live in a config file (YAML or JSON) so they can be updated without touching code.

### Phase 4 — Adult / Juvenile classification

The `life_stage` column is already in the detection output. Right now it's based on a single length threshold. A better approach:

- Species-specific thresholds (a GoldFish adult is much smaller than a RibbonedSweetlips adult)
- Optionally, a second classifier head trained on body shape

### Phase 5 — OAK-D Pro integration

Key changes needed:

1. Replace the Streamlit file uploader with a DepthAI pipeline that streams frames from the OAK-D.
2. Extract depth at the centre of each bounding box to get real-world distance.
3. Convert pixel width + distance → physical length in cm.
4. The existing `predictor.py` and `visualization.py` don't need to change at all — the depth pipeline just feeds a different data source.

### Phase 6 — Real-time monitoring dashboard

- Per-species count over time (time-series chart)
- Alert system (e.g., "species X has not been seen for N minutes")
- Data logging to SQLite or CSV
- Possibly a separate FastAPI backend with a Streamlit or React front-end

---

## Data pipeline

```
Roboflow dataset (CC BY 4.0)
        │
        ▼
  data_sets/
  ├── train/  (6,842 images)
  ├── valid/  (700 images)
  └── test/   (700 images)
        │
        ▼
  yolo detect train --data data.yaml
        │
        ▼
  models/best.pt  (~6 MB)
        │
        ▼
  run_inference() in src/detector.py
        │
        ▼
  detections_to_dataframe() in src/predictor.py
        │
        ▼
  Streamlit UI (app.py + src/visualization.py)
```

---

## Known limitations

- **Single image only.** No video or real-time stream support yet.
- **Length estimation is approximate.** The `cm_per_pixel` scale assumes the camera is perpendicular to the fish and at a fixed distance. Depth variation causes error.
- **YOLOv8n is small.** If accuracy matters more than speed, consider YOLOv8s or YOLOv8m.
- **No CI/CD.** No automated tests or deployment pipeline yet.
