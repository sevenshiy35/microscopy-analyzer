"""
Traditional image-processing utilities for fluorescence microscopy analysis.

This module currently implements blue-channel nucleus detection. The functions
are intentionally channel-aware so green cytoskeleton and red protein intensity
workflows can be added later without changing the Streamlit app structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence


ThresholdMode = Literal["Otsu", "Manual"]


@dataclass(frozen=True)
class NucleusAnalysisConfig:
    """User-configurable parameters for blue-channel nucleus segmentation."""

    threshold_mode: ThresholdMode = "Otsu"
    manual_threshold: int = 60
    gaussian_blur_kernel: int = 5
    morph_close_kernel_size: int = 7
    min_nucleus_area: int = 1000
    exclude_edge_nuclei: bool = True
    edge_margin: int = 5


def load_uploaded_image_as_rgb(pil_image: Image.Image) -> np.ndarray:
    """
    Convert an uploaded PIL image into an RGB uint8 NumPy array.

    Streamlit uploads may be PNG/JPEG/TIFF and TIFF data can be 16-bit. This
    function normalizes non-8-bit inputs for display and thresholding.
    """

    first_frame = next(ImageSequence.Iterator(pil_image))
    image_array = np.asarray(first_frame)

    if image_array.ndim == 2:
        image_array = normalize_to_uint8(image_array)
        return np.stack([image_array, image_array, image_array], axis=-1)

    if image_array.ndim == 3 and image_array.shape[2] >= 3:
        rgb_array = image_array[:, :, :3]
        return normalize_to_uint8(rgb_array)

    # Fallback for uncommon PIL modes such as palette images.
    return np.asarray(first_frame.convert("RGB"), dtype=np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize image data to uint8.

    8-bit images pass through unchanged. Integer or floating-point arrays with
    wider ranges are linearly scaled to 0-255, which is suitable for this
    display-oriented prototype.
    """

    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()

    array_float = array.astype(np.float32)
    finite_mask = np.isfinite(array_float)
    if not finite_mask.any():
        return np.zeros(array.shape, dtype=np.uint8)

    min_value = float(array_float[finite_mask].min())
    max_value = float(array_float[finite_mask].max())

    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array_float - min_value) / (max_value - min_value)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def ensure_odd_kernel_size(kernel_size: int, fallback: int = 3) -> int:
    """Return a valid positive odd kernel size for OpenCV operations."""

    if kernel_size <= 0:
        return fallback
    if kernel_size % 2 == 0:
        return kernel_size + 1
    return kernel_size


def extract_channel(rgb_image: np.ndarray, channel: Literal["red", "green", "blue"]) -> np.ndarray:
    """Extract a single color channel from an RGB image."""

    channel_indices = {"red": 0, "green": 1, "blue": 2}
    return rgb_image[:, :, channel_indices[channel]].copy()


def threshold_channel(
    channel_image: np.ndarray,
    mode: ThresholdMode,
    manual_threshold: int,
) -> tuple[np.ndarray, float]:
    """
    Segment a grayscale channel with Otsu or manual thresholding.

    Returns the binary mask and the threshold value that was used.
    """

    if mode == "Otsu":
        threshold_value, binary_mask = cv2.threshold(
            channel_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary_mask, float(threshold_value)

    threshold_value = int(np.clip(manual_threshold, 0, 255))
    _, binary_mask = cv2.threshold(
        channel_image,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )
    return binary_mask, float(threshold_value)


def clean_binary_mask(binary_mask: np.ndarray, close_kernel_size: int) -> np.ndarray:
    """
    Fill small holes and remove isolated noise from a binary mask.

    Morphological close is used first to connect fragmented nuclear regions,
    followed by a small open operation to remove speckles.
    """

    close_size = ensure_odd_kernel_size(close_kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    opened_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, opened_kernel)
    return cleaned


def touches_or_near_edge(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    edge_margin: int,
) -> bool:
    """Check whether a component touches or lies near the image boundary."""

    margin = max(0, int(edge_margin))
    return (
        x <= margin
        or y <= margin
        or (x + width) >= (image_width - margin)
        or (y + height) >= (image_height - margin)
    )


def analyze_blue_nuclei(
    rgb_image: np.ndarray,
    config: NucleusAnalysisConfig,
) -> dict[str, np.ndarray | pd.DataFrame | float]:
    """
    Detect blue fluorescence nuclei and compute per-nucleus statistics.

    Processing flow:
    RGB image -> blue channel -> Gaussian blur -> threshold -> morphology ->
    connected components -> area/edge filtering -> overlay and table outputs.
    """

    rgb_uint8 = normalize_to_uint8(rgb_image)
    blue_channel = extract_channel(rgb_uint8, "blue")

    blur_size = ensure_odd_kernel_size(config.gaussian_blur_kernel)
    blurred_blue = cv2.GaussianBlur(blue_channel, (blur_size, blur_size), 0)

    binary_mask, threshold_value = threshold_channel(
        blurred_blue,
        config.threshold_mode,
        config.manual_threshold,
    )
    cleaned_mask = clean_binary_mask(binary_mask, config.morph_close_kernel_size)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned_mask,
        connectivity=8,
    )

    overlay_rgb = rgb_uint8.copy()
    accepted_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
    image_height, image_width = cleaned_mask.shape
    records: list[dict[str, float | int | bool]] = []
    accepted_id = 1

    for component_label in range(1, component_count):
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = centroids[component_label]

        if area < config.min_nucleus_area:
            continue

        is_edge_nucleus = touches_or_near_edge(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            edge_margin=config.edge_margin,
        )

        if config.exclude_edge_nuclei and is_edge_nucleus:
            continue

        component_mask = (labels == component_label).astype(np.uint8) * 255
        accepted_mask[labels == component_label] = 255

        mean_blue = float(cv2.mean(blue_channel, mask=component_mask)[0])
        max_blue = int(blue_channel[labels == component_label].max())
        integrated_blue = float(blue_channel[labels == component_label].sum())

        records.append(
            {
                "Nucleus_ID": accepted_id,
                "Area_px": area,
                "Bounding_Box_X": x,
                "Bounding_Box_Y": y,
                "Bounding_Box_W": width,
                "Bounding_Box_H": height,
                "Centroid_X": round(float(centroid_x), 2),
                "Centroid_Y": round(float(centroid_y), 2),
                "Mean_Blue_Intensity": round(mean_blue, 2),
                "Max_Blue_Intensity": max_blue,
                "Integrated_Blue_Intensity": round(integrated_blue, 2),
                "Is_Edge_Nucleus": bool(is_edge_nucleus),
            }
        )

        draw_component_annotation(
            overlay_rgb=overlay_rgb,
            component_mask=component_mask,
            nucleus_id=accepted_id,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )
        accepted_id += 1

    statistics = pd.DataFrame.from_records(records, columns=statistics_columns())

    return {
        "original_rgb": rgb_uint8,
        "blue_channel": blue_channel,
        "binary_mask": binary_mask,
        "cleaned_mask": accepted_mask,
        "overlay_rgb": overlay_rgb,
        "statistics": statistics,
        "threshold_value": threshold_value,
    }


def draw_component_annotation(
    overlay_rgb: np.ndarray,
    component_mask: np.ndarray,
    nucleus_id: int,
    centroid_x: float,
    centroid_y: float,
) -> None:
    """
    Draw green contours and ID labels for one detected nucleus.

    OpenCV drawing functions expect BGR colors conceptually, but arrays are just
    numeric channels. Because overlay_rgb is RGB, green remains (0, 255, 0).
    """

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

    text_position = (int(round(centroid_x)), int(round(centroid_y)))
    cv2.putText(
        overlay_rgb,
        str(nucleus_id),
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def statistics_columns() -> list[str]:
    """Central definition of the statistics table schema."""

    return [
        "Nucleus_ID",
        "Area_px",
        "Bounding_Box_X",
        "Bounding_Box_Y",
        "Bounding_Box_W",
        "Bounding_Box_H",
        "Centroid_X",
        "Centroid_Y",
        "Mean_Blue_Intensity",
        "Max_Blue_Intensity",
        "Integrated_Blue_Intensity",
        "Is_Edge_Nucleus",
    ]


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a statistics table to UTF-8 CSV bytes."""

    return dataframe.to_csv(index=False).encode("utf-8")


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Serialize a grayscale or RGB uint8 NumPy image to PNG bytes."""

    uint8_image = normalize_to_uint8(image)
    pil_image = Image.fromarray(uint8_image)
    output = BytesIO()
    pil_image.save(output, format="PNG")
    return output.getvalue()
