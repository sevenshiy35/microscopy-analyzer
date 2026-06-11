"""
Streamlit interface for the Microscopy Fluorescence Image Analyzer.

The first prototype focuses on blue fluorescence nuclei detection. The image
processing code lives in image_analysis.py so future green/red channel analyses
can be added without crowding the UI layer.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from image_analysis import (
    NucleusAnalysisConfig,
    analyze_blue_nuclei,
    analyze_green_cytoskeleton_area,
    dataframe_to_csv_bytes,
    image_to_png_bytes,
    load_uploaded_image_as_rgb,
)


st.set_page_config(
    page_title="Microscopy Fluorescence Image Analyzer",
    page_icon="M",
    layout="wide",
)


def build_sidebar_config() -> NucleusAnalysisConfig:
    """Collect user-adjustable segmentation parameters from the sidebar."""

    st.sidebar.header("Analysis Parameters")

    threshold_mode = st.sidebar.radio(
        "Blue channel threshold mode",
        options=["Otsu", "Manual"],
        index=0,
    )

    manual_threshold = st.sidebar.slider(
        "Manual threshold slider",
        min_value=0,
        max_value=255,
        value=60,
        step=1,
        help="Used only when threshold mode is set to Manual.",
    )

    gaussian_blur_kernel = st.sidebar.selectbox(
        "Gaussian blur kernel",
        options=[3, 5, 7, 9],
        index=1,
    )

    morph_close_kernel_size = st.sidebar.selectbox(
        "Morph close kernel size",
        options=[3, 5, 7, 9, 11],
        index=2,
    )

    min_nucleus_area = st.sidebar.number_input(
        "Minimum nucleus area",
        min_value=1,
        value=1000,
        step=50,
        help="Candidate regions smaller than this value are removed.",
    )

    exclude_edge_nuclei = st.sidebar.checkbox(
        "Exclude edge nuclei",
        value=True,
        help="Remove nuclei touching or too close to the image boundary.",
    )

    edge_margin = st.sidebar.number_input(
        "Edge margin",
        min_value=0,
        value=5,
        step=1,
        help="Distance in pixels used when excluding edge nuclei.",
    )

    return NucleusAnalysisConfig(
        threshold_mode=threshold_mode,
        manual_threshold=manual_threshold,
        gaussian_blur_kernel=gaussian_blur_kernel,
        morph_close_kernel_size=morph_close_kernel_size,
        min_nucleus_area=int(min_nucleus_area),
        exclude_edge_nuclei=exclude_edge_nuclei,
        edge_margin=int(edge_margin),
    )


def show_metric_row(stats_df: pd.DataFrame) -> None:
    """Display top-level summary metrics."""

    count = len(stats_df)
    mean_area = float(stats_df["Area_px"].mean()) if count else 0.0
    mean_intensity = float(stats_df["Mean_Blue_Intensity"].mean()) if count else 0.0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Detected nuclei count", f"{count}")
    metric_cols[1].metric("Mean nucleus area", f"{mean_area:.1f} px")
    metric_cols[2].metric("Mean blue intensity", f"{mean_intensity:.2f}")


def show_image_grid(results: dict) -> None:
    """Render the requested image views in a compact grid."""

    row1 = st.columns(2)
    row1[0].image(results["original_rgb"], caption="Original Image", use_container_width=True)
    row1[1].image(results["blue_channel"], caption="Blue Channel", use_container_width=True)

    row2 = st.columns(2)
    row2[0].image(results["binary_mask"], caption="Binary Mask", use_container_width=True)
    row2[1].image(results["cleaned_mask"], caption="Cleaned Mask", use_container_width=True)

    st.image(
        results["overlay_rgb"],
        caption="Overlay with Detected Nuclei",
        use_container_width=True,
    )


def show_downloads(stats_df: pd.DataFrame, results: dict) -> None:
    """Expose downloadable CSV and PNG outputs."""

    st.subheader("Download Results")
    download_cols = st.columns(3)

    download_cols[0].download_button(
        label="Download CSV statistics",
        data=dataframe_to_csv_bytes(stats_df),
        file_name="nuclei_statistics.csv",
        mime="text/csv",
        disabled=stats_df.empty,
    )

    download_cols[1].download_button(
        label="Download overlay PNG",
        data=image_to_png_bytes(results["overlay_rgb"]),
        file_name="detected_nuclei_overlay.png",
        mime="image/png",
    )

    download_cols[2].download_button(
        label="Download binary mask PNG",
        data=image_to_png_bytes(results["cleaned_mask"]),
        file_name="nuclei_binary_mask.png",
        mime="image/png",
    )


def build_green_parameter_panel() -> dict:
    """Collect green cytoskeleton analysis parameters inside the green tab."""

    st.subheader("Green Cytoskeleton Parameters")
    analysis_target = st.radio(
        "Analysis target",
        options=["Cytoskeleton fibers only", "Estimated cell-covered area"],
        index=1,
        key="green_analysis_target",
        horizontal=True,
    )
    st.caption(
        "Cytoskeleton fibers only = detects fluorescent cytoskeleton pixels. "
        "Estimated cell-covered area = estimates the larger cell-covered regions "
        "enclosed or supported by cytoskeleton networks."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        threshold_mode = st.radio(
            "Green threshold mode",
            options=["Otsu", "Manual"],
            index=1,
            key="green_threshold_mode",
            horizontal=True,
        )
        manual_threshold = st.slider(
            "Manual green threshold slider",
            min_value=0,
            max_value=255,
            value=25,
            step=1,
            key="green_manual_threshold",
            help="Used only when green threshold mode is set to Manual.",
        )
        gaussian_kernel = st.selectbox(
            "Green Gaussian blur kernel",
            options=[3, 5, 7],
            index=0,
            key="green_gaussian_kernel",
        )
        use_background_subtraction = st.checkbox(
            "Background subtraction",
            value=True,
            key="green_use_background_subtraction",
            help="Approximated with a large Gaussian blur background subtraction.",
        )
        background_kernel = st.selectbox(
            "Background blur kernel",
            options=[31, 51, 71, 101],
            index=2,
            key="green_background_kernel",
        )
        morph_open_kernel = st.selectbox(
            "Morph open kernel size",
            options=[0, 3, 5],
            index=0,
            key="green_morph_open_kernel",
            help="Use 0 to skip opening so weak fine fibers are preserved.",
        )
        fiber_close_kernel = st.selectbox(
            "Fiber close kernel size",
            options=[3, 5, 7, 9],
            index=1,
            key="green_fiber_close_kernel",
        )

    with right_col:
        area_dilation_kernel = st.selectbox(
            "Area estimation dilation kernel",
            options=[5, 9, 13, 17, 21, 25],
            index=3,
            key="green_area_dilation_kernel",
        )
        area_close_kernel = st.selectbox(
            "Area estimation close kernel",
            options=[15, 25, 35, 45, 55],
            index=2,
            key="green_area_close_kernel",
        )
        fill_holes = st.checkbox(
            "Fill holes",
            value=True,
            key="green_fill_holes",
        )
        area_smoothing_kernel = st.selectbox(
            "Area smoothing kernel",
            options=[0, 9, 15, 21, 31],
            index=2,
            key="green_area_smoothing_kernel",
        )
        min_area = st.number_input(
            "Minimum estimated area",
            min_value=1,
            value=20000,
            step=500,
            key="green_min_area",
        )
        exclude_edge_regions = st.checkbox(
            "Exclude edge regions",
            value=False,
            key="green_exclude_edge_regions",
        )
        edge_margin = st.number_input(
            "Green edge margin",
            min_value=0,
            value=5,
            step=1,
            key="green_edge_margin",
        )

    return {
        "analysis_target": analysis_target,
        "threshold_mode": threshold_mode,
        "manual_threshold": manual_threshold,
        "gaussian_kernel": gaussian_kernel,
        "use_background_subtraction": use_background_subtraction,
        "background_kernel": background_kernel,
        "morph_open_kernel": morph_open_kernel,
        "fiber_close_kernel": fiber_close_kernel,
        "area_dilation_kernel": area_dilation_kernel,
        "area_close_kernel": area_close_kernel,
        "fill_holes": fill_holes,
        "area_smoothing_kernel": area_smoothing_kernel,
        "min_area": int(min_area),
        "exclude_edge_regions": exclude_edge_regions,
        "edge_margin": int(edge_margin),
    }


def show_green_metric_rows(summary_metrics: dict) -> None:
    """Display top-level green cytoskeleton summary metrics."""

    first_row = st.columns(3)
    first_row[0].metric(
        "Estimated cell-covered region count",
        f"{summary_metrics['regions_count']}",
    )
    first_row[1].metric(
        "Total estimated cell-covered area px",
        f"{summary_metrics['total_estimated_cell_area_px']}",
    )
    first_row[2].metric(
        "Mean estimated cell-covered area px",
        f"{summary_metrics['mean_estimated_cell_area_px']:.1f}",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Total cytoskeleton fiber pixel area px",
        f"{summary_metrics['total_cytoskeleton_fiber_pixel_area_px']}",
    )
    second_row[1].metric(
        "Mean green intensity inside estimated areas",
        f"{summary_metrics['mean_green_intensity_inside_estimated_areas']:.2f}",
    )
    second_row[2].metric(
        "Cytoskeleton coverage ratio",
        f"{summary_metrics['cytoskeleton_coverage_ratio']:.4f}",
    )


def show_green_image_grid(rgb_image, green_results: dict) -> None:
    """Render green cytoskeleton analysis image outputs."""

    row1 = st.columns(2)
    row1[0].image(rgb_image, caption="Original Image", use_container_width=True)
    row1[1].image(
        green_results["green_channel_display"],
        caption="Green Channel",
        use_container_width=True,
    )

    row2 = st.columns(2)
    row2[0].image(
        green_results["green_enhanced"],
        caption="Enhanced Green Channel",
        use_container_width=True,
    )
    row2[1].image(
        green_results["cytoskeleton_fiber_mask"],
        caption="Cytoskeleton Fiber Mask",
        use_container_width=True,
    )

    st.image(
        green_results["estimated_cell_area_mask"],
        caption="Estimated Cell-covered Area Mask",
        use_container_width=True,
    )

    row3 = st.columns(2)
    row3[0].image(
        green_results["overlay_fibers"],
        caption="Overlay: Cytoskeleton Fibers",
        use_container_width=True,
    )
    row3[1].image(
        green_results["overlay_estimated_area"],
        caption="Overlay: Estimated Cell-covered Areas",
        use_container_width=True,
    )


def show_green_downloads(green_df: pd.DataFrame, green_results: dict) -> None:
    """Expose green cytoskeleton CSV and PNG downloads."""

    st.subheader("Download Green Cytoskeleton Results")
    first_row = st.columns(3)
    first_row[0].download_button(
        label="Download green cytoskeleton statistics CSV",
        data=dataframe_to_csv_bytes(green_df),
        file_name="green_cytoskeleton_statistics.csv",
        mime="text/csv",
        disabled=green_df.empty,
    )
    first_row[1].download_button(
        label="Download cytoskeleton fiber mask PNG",
        data=image_to_png_bytes(green_results["cytoskeleton_fiber_mask"]),
        file_name="green_cytoskeleton_fiber_mask.png",
        mime="image/png",
    )
    first_row[2].download_button(
        label="Download estimated cell area mask PNG",
        data=image_to_png_bytes(green_results["estimated_cell_area_mask"]),
        file_name="estimated_cell_area_mask.png",
        mime="image/png",
    )

    second_row = st.columns(2)
    second_row[0].download_button(
        label="Download overlay fibers PNG",
        data=image_to_png_bytes(green_results["overlay_fibers"]),
        file_name="green_cytoskeleton_fibers_overlay.png",
        mime="image/png",
    )
    second_row[1].download_button(
        label="Download estimated area overlay PNG",
        data=image_to_png_bytes(green_results["overlay_estimated_area"]),
        file_name="estimated_cell_area_overlay.png",
        mime="image/png",
    )


def main() -> None:
    """Run the Streamlit app."""

    st.title("Microscopy Fluorescence Image Analyzer")
    st.caption(
        "Prototype for traditional image-processing analysis of fluorescence microscopy images."
    )

    config = build_sidebar_config()

    uploaded_file = st.file_uploader(
        "Upload a fluorescence microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
    )

    blue_tab, green_tab = st.tabs(["Blue Channel Nuclei Detection", "Green Cytoskeleton Area"])

    if uploaded_file is None:
        with blue_tab:
            st.info("Upload a PNG, JPG, JPEG, TIF, or TIFF image to start blue nuclei analysis.")
        with green_tab:
            st.info(
                "Upload a PNG, JPG, JPEG, TIF, or TIFF image to start green cytoskeleton area analysis."
            )
        return

    try:
        image_bytes = BytesIO(uploaded_file.getvalue())
        pil_image = Image.open(image_bytes)
        rgb_image = load_uploaded_image_as_rgb(pil_image)
    except Exception as exc:
        st.error(f"Could not read image file: {exc}")
        return

    with blue_tab:
        with st.spinner("Analyzing blue fluorescence nuclei..."):
            results = analyze_blue_nuclei(rgb_image, config)

        stats_df = results["statistics"]

        show_metric_row(stats_df)
        st.divider()
        show_image_grid(results)

        st.subheader("Nucleus Statistics")
        if stats_df.empty:
            st.warning("No nuclei were detected with the current parameters.")
        else:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        show_downloads(stats_df, results)

    with green_tab:
        green_params = build_green_parameter_panel()
        if green_params["analysis_target"] == "Cytoskeleton fibers only":
            st.info(
                "Fiber-only view highlights fluorescent cytoskeleton pixels. "
                "Individual fiber fragments are not assigned Region_ID labels."
            )
        else:
            st.info(
                "Estimated area view assigns Region_ID labels only to large cell-covered regions "
                "derived from the cytoskeleton network."
            )

        with st.spinner("Analyzing green cytoskeleton area..."):
            green_results = analyze_green_cytoskeleton_area(
                rgb_image,
                threshold_mode=green_params["threshold_mode"],
                manual_threshold=green_params["manual_threshold"],
                gaussian_kernel=green_params["gaussian_kernel"],
                use_background_subtraction=green_params["use_background_subtraction"],
                background_kernel=green_params["background_kernel"],
                morph_open_kernel=green_params["morph_open_kernel"],
                fiber_close_kernel=green_params["fiber_close_kernel"],
                area_dilation_kernel=green_params["area_dilation_kernel"],
                area_close_kernel=green_params["area_close_kernel"],
                fill_holes=green_params["fill_holes"],
                area_smoothing_kernel=green_params["area_smoothing_kernel"],
                min_area=green_params["min_area"],
                exclude_edge_regions=green_params["exclude_edge_regions"],
                edge_margin=green_params["edge_margin"],
            )

        green_df = green_results["results_dataframe"]
        st.caption(
            "Region IDs and statistics represent estimated cell-covered areas. "
            "The cytoskeleton fiber mask is shown for fluorescence-pixel inspection and coverage metrics."
        )
        show_green_metric_rows(green_results["summary_metrics"])
        st.divider()
        show_green_image_grid(rgb_image, green_results)

        st.subheader("Green Cytoskeleton Area Statistics")
        if green_df.empty:
            st.warning("No green cell-covered regions were detected with the current parameters.")
        else:
            st.dataframe(green_df, use_container_width=True, hide_index=True)

        show_green_downloads(green_df, green_results)


if __name__ == "__main__":
    main()
