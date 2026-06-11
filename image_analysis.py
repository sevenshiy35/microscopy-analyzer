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


def normalize_mode_name(mode: str) -> str:
    """Normalize UI option text for analysis functions."""

    return mode.strip().lower().replace("-", "_").replace(" ", "_")


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


def clean_cytoskeleton_mask(
    binary_mask: np.ndarray,
    open_kernel_size: int,
    close_kernel_size: int,
) -> np.ndarray:
    """
    Create a cytoskeleton fiber mask without turning fibers into regions.

    Opening is optional because fine actin-like fibers can be only a few pixels
    wide. Closing is intentionally small and only repairs short breaks.
    """

    close_size = ensure_odd_kernel_size(close_kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    cleaned = binary_mask.copy()
    if open_kernel_size > 0:
        open_size = ensure_odd_kernel_size(open_kernel_size)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def estimate_cell_area_from_fibers(
    cytoskeleton_fiber_mask: np.ndarray,
    dilation_kernel_size: int,
    close_kernel_size: int,
    fill_holes: bool,
    smoothing_kernel_size: int,
) -> np.ndarray:
    """
    Convert fine cytoskeleton fibers into a broader estimated cell area mask.

    This is not single-cell segmentation. It is a traditional CV approximation
    that thickens fibers, connects nearby networks, fills enclosed gaps, and
    smooths the resulting covered-area mask.
    """

    dilation_size = ensure_odd_kernel_size(dilation_kernel_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    area_mask = cv2.dilate(cytoskeleton_fiber_mask, dilation_kernel, iterations=1)

    close_size = ensure_odd_kernel_size(close_kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    area_mask = cv2.morphologyEx(area_mask, cv2.MORPH_CLOSE, close_kernel)

    if fill_holes:
        area_mask = fill_binary_holes(area_mask)

    if smoothing_kernel_size > 0:
        smooth_size = ensure_odd_kernel_size(smoothing_kernel_size)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_size, smooth_size))
        area_mask = cv2.morphologyEx(area_mask, cv2.MORPH_CLOSE, smooth_kernel)
        area_mask = cv2.morphologyEx(area_mask, cv2.MORPH_OPEN, smooth_kernel)

    return (area_mask > 0).astype(np.uint8) * 255


def create_weak_green_mask(green_channel: np.ndarray, weak_green_threshold: int) -> np.ndarray:
    """
    Detect dim green evidence using CLAHE on the original green channel.

    CLAHE improves faint local texture without pulling in all weak background;
    the resulting mask is still gated later by strong-fiber support or holes.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    threshold_value = int(np.clip(weak_green_threshold, 0, 255))
    _, clahe_mask = cv2.threshold(enhanced_green, threshold_value, 255, cv2.THRESH_BINARY)
    original_floor = max(3, threshold_value // 2)
    _, original_mask = cv2.threshold(green_channel, original_floor, 255, cv2.THRESH_BINARY)
    weak_mask = cv2.bitwise_and(clahe_mask, original_mask)
    return weak_mask


def rescue_weak_signal_near_fibers(
    weak_green_mask: np.ndarray,
    strong_fiber_mask: np.ndarray,
    connection_radius: int,
) -> np.ndarray:
    """
    Keep weak green pixels only when they are near clear cytoskeleton fibers.

    This prevents low-threshold background pixels far from real structures from
    being added to the area-estimation seed.
    """

    radius = ensure_odd_kernel_size(connection_radius)
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    strong_support_zone = cv2.dilate(strong_fiber_mask, support_kernel, iterations=1)
    rescued = ((weak_green_mask > 0) & (strong_support_zone > 0)).astype(np.uint8) * 255
    return rescued


def identify_binary_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Return enclosed holes in a binary mask without filling edge background."""

    mask = (binary_mask > 0).astype(np.uint8) * 255
    flood = mask.copy()
    height, width = flood.shape
    flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    border_points: list[tuple[int, int]] = []
    for x_coord in range(width):
        border_points.append((x_coord, 0))
        border_points.append((x_coord, height - 1))
    for y_coord in range(height):
        border_points.append((0, y_coord))
        border_points.append((width - 1, y_coord))

    for seed_x, seed_y in border_points:
        if flood[seed_y, seed_x] == 0:
            cv2.floodFill(flood, flood_fill_mask, (seed_x, seed_y), 255)

    return (flood == 0).astype(np.uint8) * 255


def selectively_fill_weak_green_holes(
    area_mask: np.ndarray,
    green_channel: np.ndarray,
    weak_green_mask: np.ndarray,
    min_weak_coverage: float,
    max_hole_area: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Fill only enclosed holes that contain measurable weak green evidence.

    Large holes, edge-connected background, and low-coverage dark holes are kept
    as holes. Returns the updated area mask, rescued-hole mask, and coverage
    ratios for rescued holes.
    """

    holes_mask = identify_binary_holes(area_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)
    filled_mask = area_mask.copy()
    rescued_hole_mask = np.zeros_like(area_mask)
    rescued_coverages: list[float] = []
    min_coverage = float(np.clip(min_weak_coverage, 0.0, 1.0))
    max_area = max(0, int(max_hole_area))
    image_height, image_width = area_mask.shape

    for component_label in range(1, component_count):
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        hole_area = int(stats[component_label, cv2.CC_STAT_AREA])

        if hole_area == 0 or hole_area > max_area:
            continue

        touches_edge = touches_or_near_edge(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            edge_margin=0,
        )
        if touches_edge:
            continue

        hole_pixels = labels == component_label
        weak_pixel_count = int(((weak_green_mask > 0) & hole_pixels).sum())
        weak_coverage = float(weak_pixel_count / hole_area)
        # Keep the mean intensity calculation explicit for future QC exports.
        _mean_green_intensity = float(green_channel[hole_pixels].mean())

        if weak_coverage >= min_coverage:
            filled_mask[hole_pixels] = 255
            rescued_hole_mask[hole_pixels] = 255
            rescued_coverages.append(weak_coverage)

    return filled_mask, rescued_hole_mask, rescued_coverages


def fill_binary_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask using OpenCV flood fill.

    The method marks background connected to image borders, then treats the
    remaining background pixels as enclosed holes. This avoids adding SciPy.
    """

    mask = (binary_mask > 0).astype(np.uint8) * 255
    flood = mask.copy()
    height, width = flood.shape
    flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    border_points: list[tuple[int, int]] = []
    for x_coord in range(width):
        border_points.append((x_coord, 0))
        border_points.append((x_coord, height - 1))
    for y_coord in range(height):
        border_points.append((0, y_coord))
        border_points.append((width - 1, y_coord))

    for seed_x, seed_y in border_points:
        if flood[seed_y, seed_x] == 0:
            cv2.floodFill(flood, flood_fill_mask, (seed_x, seed_y), 255)

    holes = flood == 0
    filled = mask.copy()
    filled[holes] = 255
    return filled


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


def analyze_green_cytoskeleton_area(
    image_rgb: np.ndarray,
    threshold_mode: str = "manual",
    manual_threshold: int = 20,
    gaussian_kernel: int = 3,
    use_background_subtraction: bool = True,
    background_kernel: int = 51,
    morph_open_kernel: int = 0,
    fiber_close_kernel: int = 5,
    area_dilation_kernel: int = 13,
    area_close_kernel: int = 35,
    fill_holes: bool = False,
    area_smoothing_kernel: int = 9,
    min_area: int = 15000,
    exclude_edge_regions: bool = False,
    edge_margin: int = 5,
    enable_weak_green_rescue: bool = True,
    weak_green_threshold: int = 15,
    weak_signal_connection_radius: int = 13,
    minimum_weak_coverage_inside_hole: float = 0.10,
    max_hole_area_to_rescue: int = 50000,
    fill_only_holes_with_green_evidence: bool = True,
) -> dict[str, np.ndarray | pd.DataFrame | dict[str, float | int]]:
    """
    Detect green cytoskeleton fibers and estimate larger cell-covered areas.

    The fiber mask represents fluorescent cytoskeleton pixels. Region IDs and
    table rows are created only from the larger estimated cell-covered area mask.
    """

    rgb_uint8 = normalize_to_uint8(image_rgb)
    green_channel = extract_channel(rgb_uint8, "green")

    if use_background_subtraction:
        background_size = ensure_odd_kernel_size(background_kernel)
        background = cv2.GaussianBlur(green_channel, (background_size, background_size), 0)
        green_enhanced_raw = cv2.subtract(green_channel, background)
        green_enhanced = normalize_to_uint8(green_enhanced_raw)
    else:
        green_enhanced = green_channel.copy()

    blur_size = ensure_odd_kernel_size(gaussian_kernel)
    blurred_green = cv2.GaussianBlur(green_enhanced, (blur_size, blur_size), 0)

    normalized_threshold_mode = threshold_mode.strip().lower()
    cv_threshold_mode: ThresholdMode = "Otsu" if normalized_threshold_mode == "otsu" else "Manual"
    raw_fiber_mask, threshold_value = threshold_channel(
        blurred_green,
        cv_threshold_mode,
        manual_threshold,
    )

    cytoskeleton_fiber_mask = clean_cytoskeleton_mask(
        raw_fiber_mask,
        open_kernel_size=morph_open_kernel,
        close_kernel_size=fiber_close_kernel,
    )

    if enable_weak_green_rescue:
        weak_green_mask = create_weak_green_mask(green_channel, weak_green_threshold)
        rescued_weak_mask = rescue_weak_signal_near_fibers(
            weak_green_mask=weak_green_mask,
            strong_fiber_mask=cytoskeleton_fiber_mask,
            connection_radius=weak_signal_connection_radius,
        )
        area_seed_mask = cv2.bitwise_or(cytoskeleton_fiber_mask, rescued_weak_mask)
    else:
        weak_green_mask = np.zeros_like(cytoskeleton_fiber_mask)
        rescued_weak_mask = np.zeros_like(cytoskeleton_fiber_mask)
        area_seed_mask = cytoskeleton_fiber_mask.copy()

    estimated_area_mask = estimate_cell_area_from_fibers(
        cytoskeleton_fiber_mask=area_seed_mask,
        dilation_kernel_size=area_dilation_kernel,
        close_kernel_size=area_close_kernel,
        fill_holes=fill_holes,
        smoothing_kernel_size=area_smoothing_kernel,
    )

    if enable_weak_green_rescue and fill_only_holes_with_green_evidence and not fill_holes:
        estimated_area_mask, rescued_hole_mask, rescued_hole_coverages = (
            selectively_fill_weak_green_holes(
                area_mask=estimated_area_mask,
                green_channel=green_channel,
                weak_green_mask=weak_green_mask,
                min_weak_coverage=minimum_weak_coverage_inside_hole,
                max_hole_area=max_hole_area_to_rescue,
            )
        )
    else:
        rescued_hole_mask = np.zeros_like(estimated_area_mask)
        rescued_hole_coverages: list[float] = []

    total_rescue_mask = cv2.bitwise_or(rescued_weak_mask, rescued_hole_mask)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        estimated_area_mask,
        connectivity=8,
    )

    filtered_area_mask = np.zeros_like(estimated_area_mask)
    overlay_fibers = make_cytoskeleton_overlay(rgb_uint8, cytoskeleton_fiber_mask)
    overlay_weak_rescue = make_weak_rescue_overlay(rgb_uint8, total_rescue_mask)
    overlay_estimated_area = rgb_uint8.copy()
    image_height, image_width = estimated_area_mask.shape
    records: list[dict[str, float | int | bool]] = []
    accepted_annotations: list[tuple[np.ndarray, int, float, float]] = []
    region_id = 1
    rescued_hole_component_count, rescued_hole_labels, _, _ = cv2.connectedComponentsWithStats(
        rescued_hole_mask,
        connectivity=8,
    )
    rescued_hole_coverage_by_label: dict[int, float] = {}
    for hole_label in range(1, rescued_hole_component_count):
        hole_pixels = rescued_hole_labels == hole_label
        hole_area = int(hole_pixels.sum())
        weak_count = int(((weak_green_mask > 0) & hole_pixels).sum())
        rescued_hole_coverage_by_label[hole_label] = (
            float(weak_count / hole_area) if hole_area else 0.0
        )

    for component_label in range(1, component_count):
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = centroids[component_label]

        if area < int(min_area):
            continue

        is_edge_region = touches_or_near_edge(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            edge_margin=edge_margin,
        )

        if exclude_edge_regions and is_edge_region:
            continue

        region_pixels = labels == component_label
        region_mask = region_pixels.astype(np.uint8) * 255
        fiber_pixels_inside_region = (cytoskeleton_fiber_mask > 0) & region_pixels
        fiber_pixel_area = int(fiber_pixels_inside_region.sum())
        coverage_ratio = float(fiber_pixel_area / area) if area else 0.0
        rescued_area_inside_region = int(((total_rescue_mask > 0) & region_pixels).sum())
        region_hole_labels = np.unique(rescued_hole_labels[region_pixels])
        region_hole_labels = region_hole_labels[region_hole_labels > 0]
        rescued_hole_count = int(len(region_hole_labels))
        region_rescued_hole_coverages = [
            rescued_hole_coverage_by_label[int(hole_label)]
            for hole_label in region_hole_labels
        ]
        mean_rescued_hole_coverage = (
            float(np.mean(region_rescued_hole_coverages))
            if region_rescued_hole_coverages
            else 0.0
        )

        filtered_area_mask[region_pixels] = 255

        mean_green = float(cv2.mean(green_channel, mask=region_mask)[0])
        max_green = int(green_channel[region_pixels].max())
        integrated_green = float(green_channel[region_pixels].sum())

        records.append(
            {
                "Region_ID": region_id,
                "Estimated_Cell_Area_px": area,
                "Bounding_Box_X": x,
                "Bounding_Box_Y": y,
                "Bounding_Box_W": width,
                "Bounding_Box_H": height,
                "Centroid_X": round(float(centroid_x), 2),
                "Centroid_Y": round(float(centroid_y), 2),
                "Mean_Green_Intensity_Inside_Area": round(mean_green, 2),
                "Max_Green_Intensity_Inside_Area": max_green,
                "Integrated_Green_Intensity_Inside_Area": round(integrated_green, 2),
                "Cytoskeleton_Fiber_Pixel_Area_Inside_Region": fiber_pixel_area,
                "Cytoskeleton_Coverage_Ratio": round(coverage_ratio, 4),
                "Rescued_Weak_Green_Area_px": rescued_area_inside_region,
                "Rescued_Hole_Count": rescued_hole_count,
                "Mean_Weak_Coverage_In_Rescued_Holes": round(mean_rescued_hole_coverage, 4),
                "Is_Edge_Region": bool(is_edge_region),
            }
        )

        accepted_annotations.append((region_mask, region_id, float(centroid_x), float(centroid_y)))
        region_id += 1

    overlay_estimated_area = blend_mask_color(
        overlay_estimated_area,
        filtered_area_mask,
        color_rgb=(255, 220, 0),
        alpha=0.35,
    )
    # Redraw annotations after blending so contours and IDs stay crisp.
    for region_mask, annotation_id, centroid_x, centroid_y in accepted_annotations:
        draw_area_annotation(
            overlay_rgb=overlay_estimated_area,
            region_mask=region_mask,
            region_id=annotation_id,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )

    results_dataframe = pd.DataFrame.from_records(records, columns=green_statistics_columns())
    summary_metrics = summarize_green_results(results_dataframe)
    summary_metrics["threshold_value"] = float(threshold_value)

    return {
        "green_channel_display": green_channel,
        "green_enhanced": green_enhanced,
        "raw_fiber_mask": raw_fiber_mask,
        "weak_green_mask": weak_green_mask,
        "rescued_weak_mask": rescued_weak_mask,
        "rescued_hole_mask": rescued_hole_mask,
        "cytoskeleton_fiber_mask": cytoskeleton_fiber_mask,
        "estimated_cell_area_mask": filtered_area_mask,
        "overlay_fibers": overlay_fibers,
        "overlay_weak_rescue": overlay_weak_rescue,
        "overlay_estimated_area": overlay_estimated_area,
        "results_dataframe": results_dataframe,
        "summary_metrics": summary_metrics,
    }


def analyze_red_protein_puncta(
    image_rgb: np.ndarray,
    threshold_mode: str = "manual",
    manual_threshold: int = 40,
    use_background_subtraction: bool = True,
    background_kernel: int = 51,
    enhancement_method: str = "white_tophat",
    top_hat_kernel: int = 7,
    remove_fibers: bool = True,
    fiber_removal_method: str = "both",
    line_suppression_length: int = 31,
    min_area: int = 3,
    max_area: int = 300,
    min_circularity: float = 0.35,
    max_aspect_ratio: float = 3.0,
    min_mean_intensity: float = 20,
    exclude_edge_puncta: bool = False,
    edge_margin: int = 3,
    merge_nearby_puncta: bool = False,
    merge_distance: int = 2,
) -> dict[str, np.ndarray | pd.DataFrame | dict[str, float | int]]:
    """
    Detect compact red target-protein puncta while suppressing red nanofibers.

    The detector enhances small bright blobs, optionally identifies long fiber
    background, then keeps connected components with puncta-like shape and
    intensity statistics.
    """

    rgb_uint8 = normalize_to_uint8(image_rgb)
    red_channel = extract_channel(rgb_uint8, "red")

    if use_background_subtraction:
        bg_size = ensure_odd_kernel_size(background_kernel)
        background = cv2.GaussianBlur(red_channel, (bg_size, bg_size), 0)
        red_preprocessed = normalize_to_uint8(cv2.subtract(red_channel, background))
    else:
        red_preprocessed = red_channel.copy()

    red_enhanced = enhance_red_puncta(
        red_preprocessed=red_preprocessed,
        enhancement_method=enhancement_method,
        top_hat_kernel=top_hat_kernel,
    )

    method = normalize_mode_name(fiber_removal_method)
    if remove_fibers and method in {"multi_angle_line_suppression", "both"}:
        fiber_background_mask = detect_red_fiber_background(
            red_preprocessed,
            line_suppression_length,
        )
    else:
        fiber_background_mask = np.zeros_like(red_channel)

    mode = "Otsu" if normalize_mode_name(threshold_mode) == "otsu" else "Manual"
    puncta_candidate_mask, threshold_value = threshold_channel(
        red_enhanced,
        mode,
        manual_threshold,
    )

    if remove_fibers and method in {"multi_angle_line_suppression", "both"}:
        puncta_candidate_mask[fiber_background_mask > 0] = 0

    if merge_nearby_puncta:
        merge_size = max(1, int(merge_distance))
        merge_kernel_size = ensure_odd_kernel_size(merge_size)
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (merge_kernel_size, merge_kernel_size),
        )
        puncta_candidate_mask = cv2.morphologyEx(
            puncta_candidate_mask,
            cv2.MORPH_CLOSE,
            merge_kernel,
        )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        puncta_candidate_mask,
        connectivity=8,
    )

    puncta_cleaned_mask = np.zeros_like(puncta_candidate_mask)
    overlay_puncta = rgb_uint8.copy()
    image_height, image_width = puncta_candidate_mask.shape
    records: list[dict[str, float | int | bool]] = []
    puncta_id = 1

    for component_label in range(1, component_count):
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = centroids[component_label]

        if area < int(min_area) or area > int(max_area):
            continue

        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        component_mask = (labels == component_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter else 0.0

        overlaps_fiber = bool(np.any((fiber_background_mask > 0) & (labels == component_label)))
        is_edge_puncta = touches_or_near_edge(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            edge_margin=edge_margin,
        )

        mean_red = float(cv2.mean(red_channel, mask=component_mask)[0])
        max_red = int(red_channel[labels == component_label].max())
        integrated_red = float(red_channel[labels == component_label].sum())

        shape_filter_active = remove_fibers and method in {"shape_filtering_only", "both"}
        if shape_filter_active:
            if circularity < float(min_circularity) or aspect_ratio > float(max_aspect_ratio):
                continue
        else:
            if circularity < float(min_circularity):
                continue

        if mean_red < float(min_mean_intensity):
            continue
        if remove_fibers and overlaps_fiber:
            continue
        if exclude_edge_puncta and is_edge_puncta:
            continue

        puncta_cleaned_mask[labels == component_label] = 255
        radius = int(np.ceil(np.sqrt(area / np.pi))) + 2
        draw_puncta_annotation(overlay_puncta, centroid_x, centroid_y, radius, puncta_id)

        records.append(
            {
                "Puncta_ID": puncta_id,
                "Area_px": area,
                "Centroid_X": round(float(centroid_x), 2),
                "Centroid_Y": round(float(centroid_y), 2),
                "Bounding_Box_X": x,
                "Bounding_Box_Y": y,
                "Bounding_Box_W": width,
                "Bounding_Box_H": height,
                "Mean_Red_Intensity": round(mean_red, 2),
                "Max_Red_Intensity": max_red,
                "Integrated_Red_Intensity": round(integrated_red, 2),
                "Circularity": round(circularity, 4),
                "Aspect_Ratio": round(aspect_ratio, 4),
                "Overlaps_Fiber_Background": overlaps_fiber,
                "Is_Edge_Puncta": bool(is_edge_puncta),
            }
        )
        puncta_id += 1

    results_dataframe = pd.DataFrame.from_records(records, columns=red_statistics_columns())
    summary_metrics = summarize_red_results(results_dataframe, fiber_background_mask)
    summary_metrics["threshold_value"] = float(threshold_value)

    return {
        "red_channel_display": red_channel,
        "red_enhanced": red_enhanced,
        "fiber_background_mask": fiber_background_mask,
        "puncta_candidate_mask": puncta_candidate_mask,
        "puncta_cleaned_mask": puncta_cleaned_mask,
        "overlay_puncta": overlay_puncta,
        "results_dataframe": results_dataframe,
        "summary_metrics": summary_metrics,
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


def blend_mask_color(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend a solid RGB color into pixels selected by a binary mask."""

    output = image_rgb.copy()
    selected_pixels = mask > 0
    if not selected_pixels.any():
        return output

    color_array = np.array(color_rgb, dtype=np.float32)
    output_float = output.astype(np.float32)
    output_float[selected_pixels] = (
        (1.0 - alpha) * output_float[selected_pixels] + alpha * color_array
    )
    return np.clip(output_float, 0, 255).astype(np.uint8)


def make_cytoskeleton_overlay(rgb_image: np.ndarray, cytoskeleton_mask: np.ndarray) -> np.ndarray:
    """Create an RGB overlay where detected green fibers are highlighted."""

    overlay = blend_mask_color(
        rgb_image,
        cytoskeleton_mask,
        color_rgb=(0, 255, 80),
        alpha=0.65,
    )
    contours, _ = cv2.findContours(
        (cytoskeleton_mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
    return overlay


def make_weak_rescue_overlay(rgb_image: np.ndarray, rescue_mask: np.ndarray) -> np.ndarray:
    """Create an RGB overlay for weak green signal rescued into area estimates."""

    overlay = blend_mask_color(
        rgb_image,
        rescue_mask,
        color_rgb=(0, 220, 255),
        alpha=0.55,
    )
    contours, _ = cv2.findContours(
        (rescue_mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 255), thickness=1)
    return overlay


def create_line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """Create a binary line structuring element at a given angle."""

    size = ensure_odd_kernel_size(length)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radians = np.deg2rad(angle_degrees)
    dx = int(round(np.cos(radians) * center))
    dy = int(round(np.sin(radians) * center))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
    return kernel


def detect_red_fiber_background(red_image: np.ndarray, line_length: int) -> np.ndarray:
    """
    Detect long red nanofiber-like structures using multi-angle openings.

    Line openings respond to elongated structures and suppress compact dots.
    """

    length = ensure_odd_kernel_size(line_length)
    responses = np.zeros_like(red_image)
    for angle in [0, 30, 60, 90, 120, 150]:
        line_kernel = create_line_kernel(length, angle)
        opened = cv2.morphologyEx(red_image, cv2.MORPH_OPEN, line_kernel)
        responses = np.maximum(responses, opened)

    if int(responses.max()) == 0:
        return np.zeros_like(red_image)

    _, fiber_mask = cv2.threshold(
        responses,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return fiber_mask


def enhance_red_puncta(
    red_preprocessed: np.ndarray,
    enhancement_method: str,
    top_hat_kernel: int,
) -> np.ndarray:
    """Enhance compact red puncta before thresholding."""

    method = normalize_mode_name(enhancement_method)
    if method in {"white_tophat", "white_top_hat"}:
        kernel_size = ensure_odd_kernel_size(top_hat_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        enhanced = cv2.morphologyEx(red_preprocessed, cv2.MORPH_TOPHAT, kernel)
        return normalize_to_uint8(enhanced)

    if method in {"log_like_blob_enhancement", "log_like", "log"}:
        blur_small = cv2.GaussianBlur(red_preprocessed, (3, 3), 0)
        blur_large = cv2.GaussianBlur(red_preprocessed, (9, 9), 0)
        enhanced = cv2.subtract(blur_small, blur_large)
        return normalize_to_uint8(enhanced)

    return red_preprocessed.copy()


def draw_puncta_annotation(
    overlay_rgb: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    radius: int,
    puncta_id: int,
) -> None:
    """Draw a compact circle and ID for a detected red punctum."""

    center = (int(round(centroid_x)), int(round(centroid_y)))
    cv2.circle(overlay_rgb, center, max(3, radius), color=(0, 255, 255), thickness=1)
    cv2.putText(
        overlay_rgb,
        str(puncta_id),
        (center[0] + 3, center[1] - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(255, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def draw_area_annotation(
    overlay_rgb: np.ndarray,
    region_mask: np.ndarray,
    region_id: int,
    centroid_x: float,
    centroid_y: float,
) -> None:
    """Draw an estimated cell-covered region outline and ID label."""

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, contourIdx=-1, color=(255, 255, 0), thickness=2)

    cv2.putText(
        overlay_rgb,
        str(region_id),
        (int(round(centroid_x)), int(round(centroid_y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(255, 255, 0),
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


def green_statistics_columns() -> list[str]:
    """Central definition of the green cytoskeleton statistics table schema."""

    return [
        "Region_ID",
        "Estimated_Cell_Area_px",
        "Bounding_Box_X",
        "Bounding_Box_Y",
        "Bounding_Box_W",
        "Bounding_Box_H",
        "Centroid_X",
        "Centroid_Y",
        "Mean_Green_Intensity_Inside_Area",
        "Max_Green_Intensity_Inside_Area",
        "Integrated_Green_Intensity_Inside_Area",
        "Cytoskeleton_Fiber_Pixel_Area_Inside_Region",
        "Cytoskeleton_Coverage_Ratio",
        "Rescued_Weak_Green_Area_px",
        "Rescued_Hole_Count",
        "Mean_Weak_Coverage_In_Rescued_Holes",
        "Is_Edge_Region",
    ]


def red_statistics_columns() -> list[str]:
    """Central definition of the red protein puncta statistics table schema."""

    return [
        "Puncta_ID",
        "Area_px",
        "Centroid_X",
        "Centroid_Y",
        "Bounding_Box_X",
        "Bounding_Box_Y",
        "Bounding_Box_W",
        "Bounding_Box_H",
        "Mean_Red_Intensity",
        "Max_Red_Intensity",
        "Integrated_Red_Intensity",
        "Circularity",
        "Aspect_Ratio",
        "Overlaps_Fiber_Background",
        "Is_Edge_Puncta",
    ]


def summarize_green_results(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Compute display-ready summary metrics for green area analysis."""

    if dataframe.empty:
        return {
            "regions_count": 0,
            "total_estimated_cell_area_px": 0,
            "mean_estimated_cell_area_px": 0.0,
            "total_cytoskeleton_fiber_pixel_area_px": 0,
            "mean_green_intensity_inside_estimated_areas": 0.0,
            "cytoskeleton_coverage_ratio": 0.0,
            "rescued_weak_green_area_px": 0,
            "rescued_hole_count": 0,
            "mean_weak_coverage_of_rescued_holes": 0.0,
        }

    total_area = int(dataframe["Estimated_Cell_Area_px"].sum())
    total_fiber_area = int(dataframe["Cytoskeleton_Fiber_Pixel_Area_Inside_Region"].sum())
    rescued_hole_count = int(dataframe["Rescued_Hole_Count"].sum())
    rescued_hole_rows = dataframe[dataframe["Rescued_Hole_Count"] > 0]

    return {
        "regions_count": int(len(dataframe)),
        "total_estimated_cell_area_px": total_area,
        "mean_estimated_cell_area_px": float(dataframe["Estimated_Cell_Area_px"].mean()),
        "total_cytoskeleton_fiber_pixel_area_px": total_fiber_area,
        "mean_green_intensity_inside_estimated_areas": float(
            dataframe["Mean_Green_Intensity_Inside_Area"].mean()
        ),
        "cytoskeleton_coverage_ratio": (
            float(total_fiber_area / total_area) if total_area else 0.0
        ),
        "rescued_weak_green_area_px": int(dataframe["Rescued_Weak_Green_Area_px"].sum()),
        "rescued_hole_count": rescued_hole_count,
        "mean_weak_coverage_of_rescued_holes": (
            float(
                np.average(
                    rescued_hole_rows["Mean_Weak_Coverage_In_Rescued_Holes"],
                    weights=rescued_hole_rows["Rescued_Hole_Count"],
                )
            )
            if rescued_hole_count
            else 0.0
        ),
    }


def summarize_red_results(
    dataframe: pd.DataFrame,
    fiber_background_mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute display-ready summary metrics for red puncta analysis."""

    fiber_area = int((fiber_background_mask > 0).sum())
    if dataframe.empty:
        return {
            "puncta_count": 0,
            "total_puncta_area_px": 0,
            "mean_puncta_area_px": 0.0,
            "mean_red_intensity_in_puncta": 0.0,
            "total_integrated_red_intensity": 0.0,
            "fiber_background_pixel_area_px": fiber_area,
            "puncta_to_fiber_area_ratio": 0.0,
        }

    total_puncta_area = int(dataframe["Area_px"].sum())
    return {
        "puncta_count": int(len(dataframe)),
        "total_puncta_area_px": total_puncta_area,
        "mean_puncta_area_px": float(dataframe["Area_px"].mean()),
        "mean_red_intensity_in_puncta": float(dataframe["Mean_Red_Intensity"].mean()),
        "total_integrated_red_intensity": float(dataframe["Integrated_Red_Intensity"].sum()),
        "fiber_background_pixel_area_px": fiber_area,
        "puncta_to_fiber_area_ratio": (
            float(total_puncta_area / fiber_area) if fiber_area else 0.0
        ),
    }


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
