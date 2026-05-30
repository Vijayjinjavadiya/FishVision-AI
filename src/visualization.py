"""
visualization.py — Streamlit rendering helpers.

Every function here writes directly to the Streamlit UI.
Keeping them here means app.py stays small and readable,
and each render function can be tested or swapped independently.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predictor import DETECTION_COLUMNS


def render_metrics(total: int, species: int, avg_confidence: float) -> None:
    """Render the three top-level detection metric cards.

    Args:
        total: Total number of fish detected.
        species: Number of distinct species detected.
        avg_confidence: Mean confidence score across all detections.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Fish Detected", total)
    col2.metric("Species Detected", species)
    col3.metric(
        "Avg Confidence",
        f"{avg_confidence:.2f}" if total else "—",
    )


def render_annotated_image(annotated_bgr: object) -> None:
    """Render the YOLO-annotated image (BGR numpy array from result.plot()).

    Args:
        annotated_bgr: NumPy array in BGR format produced by
            ``result.plot()``.
    """
    st.subheader("Detection Output")
    st.image(annotated_bgr, channels="BGR", use_container_width=True)


def render_class_counts(df: pd.DataFrame) -> None:
    """Render a small table of detected species and their counts.

    Args:
        df: Detections DataFrame (output of :func:`~src.predictor.detections_to_dataframe`).
    """
    st.subheader("Species Breakdown")
    if df.empty:
        st.info("No fish detected at the selected confidence threshold.")
        return

    counts = (
        df["class"]
        .value_counts()
        .rename_axis("Species")
        .reset_index(name="Count")
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)


def render_detection_table(df: pd.DataFrame, cm_per_pixel: float) -> None:
    """Render the detailed per-fish detection table with a CSV download.

    Args:
        df: Detections DataFrame.
        cm_per_pixel: Scale factor; determines whether length/life-stage
            columns are shown.
    """
    st.subheader("Detection Details")

    visible = ["fish_id", "class", "confidence", "bbox_width_px", "bbox_height_px"]
    if cm_per_pixel > 0:
        visible.extend(["estimated_length_cm", "life_stage"])

    if df.empty:
        st.dataframe(df[[] if df.empty else visible], use_container_width=True, hide_index=True)
        return

    st.dataframe(df[visible], use_container_width=True, hide_index=True)
    st.download_button(
        label="Download detections as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fish_detections.csv",
        mime="text/csv",
    )
