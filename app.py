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
            value=35,
            step=1,
            key="green_manual_threshold",
            help="Used only when green threshold mode is set to Manual.",
        )
        gaussian_kernel = st.selectbox(
            "Green Gaussian blur kernel",
            options=[3, 5, 7, 9],
            index=1,
            key="green_gaussian_kernel",
        )
        use_background_subtraction = st.checkbox(
            "Background subtraction / rolling ball approximation",
            value=True,
            key="green_use_background_subtraction",
            help="Approximated with a large Gaussian blur background subtraction.",
        )
        background_kernel = st.selectbox(
            "Background blur kernel",
            options=[21, 31, 51, 71],
            index=2,
            key="green_background_kernel",
        )
        morph_close_kernel = st.selectbox(
            "Green morph close kernel size",
            options=[5, 9, 15, 21, 31],
            index=2,
            key="green_morph_close_kernel",
        )

    with right_col:
        dilate_before_fill = st.checkbox(
            "Dilate cytoskeleton before area fill",
            value=True,
            key="green_dilate_before_fill",
        )
        dilation_kernel = st.selectbox(
            "Dilation kernel size",
            options=[3, 5, 7, 9, 11],
            index=2,
            key="green_dilation_kernel",
        )
        fill_enclosed_regions = st.checkbox(
            "Fill enclosed regions",
            value=True,
            key="green_fill_enclosed_regions",
        )
        min_area = st.number_input(
            "Minimum cell-covered area",
            min_value=1,
            value=3000,
            step=100,
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
        "threshold_mode": threshold_mode,
        "manual_threshold": manual_threshold,
        "gaussian_kernel": gaussian_kernel,
        "use_background_subtraction": use_background_subtraction,
        "background_kernel": background_kernel,
        "morph_close_kernel": morph_close_kernel,
        "dilate_before_fill": dilate_before_fill,
        "dilation_kernel": dilation_kernel,
        "fill_enclosed_regions": fill_enclosed_regions,
        "min_area": int(min_area),
        "exclude_edge_regions": exclude_edge_regions,
        "edge_margin": int(edge_margin),
    }


def show_green_metric_rows(summary_metrics: dict) -> None:
    """Display top-level green cytoskeleton summary metrics."""

    first_row = st.columns(3)
    first_row[0].metric(
        "Estimated cell-covered regions count",
        f"{summary_metrics['regions_count']}",
    )
    first_row[1].metric(
        "Total estimated cell-covered area px",
        f"{summary_metrics['total_estimated_cell_area_px']}",
    )
    first_row[2].metric(
        "Mean estimated region area px",
        f"{summary_metrics['mean_estimated_region_area_px']:.1f}",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Total cytoskeleton pixel area px",
        f"{summary_metrics['total_cytoskeleton_pixel_area_px']}",
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
        caption="Background-subtracted / Enhanced Green Channel",
        use_container_width=True,
    )
    row2[1].image(
        green_results["cytoskeleton_binary_mask"],
        caption="Cytoskeleton Binary Mask",
        use_container_width=True,
    )

    row3 = st.columns(2)
    row3[0].image(
        green_results["cytoskeleton_cleaned_mask"],
        caption="Cleaned Cytoskeleton Mask",
        use_container_width=True,
    )
    row3[1].image(
        green_results["estimated_cell_area_mask"],
        caption="Estimated Cell-covered Area Mask",
        use_container_width=True,
    )

    row4 = st.columns(2)
    row4[0].image(
        green_results["overlay_cytoskeleton"],
        caption="Overlay: Cytoskeleton",
        use_container_width=True,
    )
    row4[1].image(
        green_results["overlay_estimated_area"],
        caption="Overlay: Estimated Cell-covered Area",
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
        label="Download cytoskeleton mask PNG",
        data=image_to_png_bytes(green_results["cytoskeleton_cleaned_mask"]),
        file_name="green_cytoskeleton_mask.png",
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
        label="Download green cytoskeleton overlay PNG",
        data=image_to_png_bytes(green_results["overlay_cytoskeleton"]),
        file_name="green_cytoskeleton_overlay.png",
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

    if uploaded_file is None:
        st.info("Upload a PNG, JPG, JPEG, TIF, or TIFF image to start analysis.")
        return

    try:
        image_bytes = BytesIO(uploaded_file.getvalue())
        pil_image = Image.open(image_bytes)
        rgb_image = load_uploaded_image_as_rgb(pil_image)
    except Exception as exc:
        st.error(f"Could not read image file: {exc}")
        return

    blue_tab, green_tab = st.tabs(["Blue Channel Nuclei Detection", "Green Cytoskeleton Area"])

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

        with st.spinner("Analyzing green cytoskeleton area..."):
            green_results = analyze_green_cytoskeleton_area(
                rgb_image,
                threshold_mode=green_params["threshold_mode"],
                manual_threshold=green_params["manual_threshold"],
                gaussian_kernel=green_params["gaussian_kernel"],
                use_background_subtraction=green_params["use_background_subtraction"],
                background_kernel=green_params["background_kernel"],
                morph_close_kernel=green_params["morph_close_kernel"],
                dilate_before_fill=green_params["dilate_before_fill"],
                dilation_kernel=green_params["dilation_kernel"],
                fill_enclosed_regions=green_params["fill_enclosed_regions"],
                min_area=green_params["min_area"],
                exclude_edge_regions=green_params["exclude_edge_regions"],
                edge_margin=green_params["edge_margin"],
            )

        green_df = green_results["results_dataframe"]
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
