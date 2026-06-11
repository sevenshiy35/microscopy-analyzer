"""
Streamlit interface for the Microscopy Fluorescence Image Analyzer.

The first prototype focuses on blue fluorescence nuclei detection. The image
processing code lives in image_analysis.py so future green/red channel analyses
can be added without crowding the UI layer.
"""

from __future__ import annotations

import hashlib
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
from parameter_config import (
    LANGUAGE_OPTIONS,
    apply_green_preset,
    apply_profile,
    build_profile,
    guide_lines,
    help_text,
    initialize_session_state,
    load_profile_bytes,
    profile_to_json_bytes,
    sanitize_profile_name,
    text,
)


st.set_page_config(
    page_title="Microscopy Fluorescence Image Analyzer",
    page_icon="M",
    layout="wide",
)


def current_language() -> str:
    """Return the selected UI language."""

    return st.session_state.get("language", "Bilingual / 双语")


def label(key: str) -> str:
    """Localized label helper."""

    return text(key, current_language())


def help_for(key: str) -> str:
    """Localized help-text helper."""

    return help_text(key, current_language())


def render_parameter_guide(module: str) -> None:
    """Show a concise localized parameter guide for the active module."""

    guide_key = "blue_guide" if module == "blue" else "green_guide"
    guide_title = (
        "Parameter Guide"
        if current_language() == "English"
        else ("参数说明" if current_language() == "中文" else "Parameter Guide / 参数说明")
    )
    with st.sidebar.expander(guide_title, expanded=False):
        st.markdown(f"**{text(guide_key, current_language())}**")
        for line in guide_lines(module, current_language()):
            st.markdown(line)


def on_green_preset_change() -> None:
    """Apply non-custom green presets to stable session-state keys."""

    preset_name = st.session_state.get("green_preset", "Balanced")
    if preset_name != "Custom":
        apply_green_preset(st.session_state, preset_name)


def on_profile_upload() -> None:
    """Load a JSON parameter profile before widgets are rendered."""

    uploaded_profile = st.session_state.get("parameter_profile_uploader")
    if uploaded_profile is None:
        return

    raw_profile = uploaded_profile.getvalue()
    profile_hash = hashlib.sha256(raw_profile).hexdigest()
    if st.session_state.get("_last_loaded_profile_hash") == profile_hash:
        return

    try:
        profile = load_profile_bytes(raw_profile)
        apply_profile(st.session_state, profile)
        st.session_state["_last_loaded_profile_hash"] = profile_hash
        st.session_state["_profile_load_success"] = True
        st.session_state.pop("_profile_load_error", None)
    except Exception:
        st.session_state["_profile_load_error"] = True


def render_profile_controls() -> None:
    """Render language and JSON parameter profile controls."""

    language = st.sidebar.selectbox(
        "Language / 语言",
        options=LANGUAGE_OPTIONS,
        key="language",
        help="Choose the interface language. / 选择界面语言。",
    )

    st.sidebar.subheader(text("parameter_profile", language))
    st.sidebar.text_input(text("profile_name", language), key="profile_name")

    if st.session_state.pop("_profile_load_success", False):
        st.sidebar.success(
            "Parameter profile loaded successfully."
            if language == "English"
            else (
                "参数配置已成功载入。"
                if language == "中文"
                else "Parameter profile loaded successfully.\n\n参数配置已成功载入。"
            )
        )
    if st.session_state.pop("_profile_load_error", False):
        st.sidebar.error(
            "Invalid parameter profile JSON."
            if language == "English"
            else (
                "参数配置文件格式无效。"
                if language == "中文"
                else "Invalid parameter profile JSON.\n\n参数配置文件格式无效。"
            )
        )

    st.sidebar.file_uploader(
        text("load_parameters", language),
        type=["json"],
        key="parameter_profile_uploader",
        on_change=on_profile_upload,
    )

    profile = build_profile(st.session_state)
    safe_name = sanitize_profile_name(profile["profile_name"])
    st.sidebar.download_button(
        text("download_parameters", language),
        data=profile_to_json_bytes(profile),
        file_name=f"microscopy_parameters_{safe_name}.json",
        mime="application/json",
    )


def build_sidebar_config() -> NucleusAnalysisConfig:
    """Collect user-adjustable segmentation parameters from the sidebar."""

    with st.sidebar.expander("Blue Channel Nuclei Detection", expanded=True):
        threshold_mode = st.radio(
            label("blue_threshold_mode"),
            options=["Otsu", "Manual"],
            key="blue_threshold_mode",
            help=help_for("blue_threshold_mode"),
        )

        manual_threshold = st.slider(
            label("blue_manual_threshold"),
            min_value=0,
            max_value=255,
            step=1,
            key="blue_manual_threshold",
            help=help_for("blue_manual_threshold"),
        )

        gaussian_blur_kernel = st.selectbox(
            label("blue_gaussian_kernel"),
            options=[3, 5, 7, 9],
            key="blue_gaussian_kernel",
            help=help_for("blue_gaussian_kernel"),
        )

        morph_close_kernel_size = st.selectbox(
            label("blue_morph_close_kernel"),
            options=[3, 5, 7, 9, 11],
            key="blue_morph_close_kernel",
            help=help_for("blue_morph_close_kernel"),
        )

        min_nucleus_area = st.number_input(
            label("blue_minimum_area"),
            min_value=1,
            step=50,
            key="blue_minimum_area",
            help=help_for("blue_minimum_area"),
        )

        exclude_edge_nuclei = st.checkbox(
            label("blue_exclude_edge_nuclei"),
            key="blue_exclude_edge_nuclei",
            help=help_for("blue_exclude_edge_nuclei"),
        )

        edge_margin = st.number_input(
            label("blue_edge_margin"),
            min_value=0,
            step=1,
            key="blue_edge_margin",
            help=help_for("blue_edge_margin"),
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
    with row1[0]:
        st.image(results["original_rgb"], caption="Original Image", use_container_width=True)
        st.download_button(
            "Download Original Image PNG",
            data=image_to_png_bytes(results["original_rgb"]),
            file_name="original_image.png",
            mime="image/png",
            key="download_blue_original_image",
        )
    with row1[1]:
        st.image(results["blue_channel"], caption="Blue Channel", use_container_width=True)
        st.download_button(
            "Download Blue Channel PNG",
            data=image_to_png_bytes(results["blue_channel"]),
            file_name="blue_channel.png",
            mime="image/png",
            key="download_blue_channel",
        )

    row2 = st.columns(2)
    with row2[0]:
        st.image(results["binary_mask"], caption="Binary Mask", use_container_width=True)
        st.download_button(
            "Download Binary Mask PNG",
            data=image_to_png_bytes(results["binary_mask"]),
            file_name="nuclei_raw_binary_mask.png",
            mime="image/png",
            key="download_blue_binary_mask",
        )
    with row2[1]:
        st.image(results["cleaned_mask"], caption="Cleaned Mask", use_container_width=True)
        st.download_button(
            "Download Cleaned Mask PNG",
            data=image_to_png_bytes(results["cleaned_mask"]),
            file_name="nuclei_binary_mask.png",
            mime="image/png",
            key="download_blue_cleaned_mask",
        )

    st.image(
        results["overlay_rgb"],
        caption="Overlay with Detected Nuclei",
        use_container_width=True,
    )
    st.download_button(
        "Download Overlay PNG",
        data=image_to_png_bytes(results["overlay_rgb"]),
        file_name="detected_nuclei_overlay.png",
        mime="image/png",
        key="download_blue_overlay",
    )


def show_downloads(stats_df: pd.DataFrame) -> None:
    """Expose downloadable CSV outputs."""

    st.subheader("Download Results")
    st.download_button(
        label="Download CSV statistics",
        data=dataframe_to_csv_bytes(stats_df),
        file_name="nuclei_statistics.csv",
        mime="text/csv",
        disabled=stats_df.empty,
    )


def build_green_parameter_panel() -> dict:
    """Collect green cytoskeleton analysis parameters from the sidebar."""

    with st.sidebar.expander("Green Cytoskeleton Area", expanded=True):
        preset_name = st.selectbox(
            label("green_preset"),
            options=["Conservative", "Balanced", "Sensitive", "Custom"],
            key="green_preset",
            on_change=on_green_preset_change,
            help=(
                "Choose a starting parameter set. Custom keeps manual edits."
                if current_language() == "English"
                else (
                    "选择一组起始参数。Custom 会保留手动修改。"
                    if current_language() == "中文"
                    else "Choose a starting parameter set. Custom keeps manual edits.\n\n选择一组起始参数。Custom 会保留手动修改。"
                )
            ),
        )
        st.caption(
            "Conservative: cleaner boundaries, less over-estimation. "
            "Balanced: recommended default for cytoskeleton-supported area estimation. "
            "Sensitive: includes weaker green signals but may include more background."
        )

        analysis_target = st.radio(
            label("green_analysis_target"),
            options=["Cytoskeleton fibers only", "Estimated cell-covered area"],
            key="green_analysis_target",
            help=help_for("green_analysis_target"),
        )
        st.caption(
            "Cytoskeleton fibers only = detects fluorescent cytoskeleton pixels. "
            "Estimated cell-covered area = estimates larger cell-covered regions "
            "enclosed or supported by cytoskeleton networks."
        )

        threshold_mode = st.radio(
            label("green_threshold_mode"),
            options=["Otsu", "Manual"],
            key="green_threshold_mode",
            help=help_for("green_threshold_mode"),
        )
        manual_threshold = st.slider(
            label("green_manual_threshold"),
            min_value=0,
            max_value=255,
            step=1,
            key="green_manual_threshold",
            help=help_for("green_manual_threshold"),
        )
        gaussian_kernel = st.selectbox(
            label("green_gaussian_kernel"),
            options=[3, 5, 7],
            key="green_gaussian_kernel",
            help=help_for("green_gaussian_kernel"),
        )
        use_background_subtraction = st.checkbox(
            label("green_background_subtraction"),
            key="green_background_subtraction",
            help=help_for("green_background_subtraction"),
        )
        background_kernel = st.selectbox(
            label("green_background_blur_kernel"),
            options=[31, 51, 71, 101],
            key="green_background_blur_kernel",
            help=help_for("green_background_blur_kernel"),
        )
        morph_open_kernel = st.selectbox(
            label("green_morph_open_kernel"),
            options=[0, 3, 5],
            key="green_morph_open_kernel",
            help=help_for("green_morph_open_kernel"),
        )
        fiber_close_kernel = st.selectbox(
            label("green_fiber_close_kernel"),
            options=[3, 5, 7, 9],
            key="green_fiber_close_kernel",
            help=help_for("green_fiber_close_kernel"),
        )

        area_dilation_kernel = st.selectbox(
            label("green_area_dilation_kernel"),
            options=[5, 9, 13, 17, 21, 25],
            key="green_area_dilation_kernel",
            help=help_for("green_area_dilation_kernel"),
        )
        area_close_kernel = st.selectbox(
            label("green_area_close_kernel"),
            options=[15, 25, 35, 45, 55],
            key="green_area_close_kernel",
            help=help_for("green_area_close_kernel"),
        )
        fill_holes = st.checkbox(
            label("green_fill_holes"),
            key="green_fill_holes",
            help=help_for("green_fill_holes"),
        )
        area_smoothing_kernel = st.selectbox(
            label("green_area_smoothing_kernel"),
            options=[0, 9, 15, 21, 31],
            key="green_area_smoothing_kernel",
            help=help_for("green_area_smoothing_kernel"),
        )
        min_area = st.number_input(
            label("green_minimum_estimated_area"),
            min_value=1,
            step=500,
            key="green_minimum_estimated_area",
            help=help_for("green_minimum_estimated_area"),
        )
        exclude_edge_regions = st.checkbox(
            label("green_exclude_edge_regions"),
            key="green_exclude_edge_regions",
            help=help_for("green_exclude_edge_regions"),
        )
        edge_margin = st.number_input(
            label("green_edge_margin"),
            min_value=0,
            step=1,
            key="green_edge_margin",
            help=help_for("green_edge_margin"),
        )
        st.divider()
        enable_weak_green_rescue = st.checkbox(
            label("green_weak_green_rescue"),
            key="green_weak_green_rescue",
            help=help_for("green_weak_green_rescue"),
        )
        weak_green_threshold = st.slider(
            label("green_weak_green_threshold"),
            min_value=0,
            max_value=255,
            step=1,
            key="green_weak_green_threshold",
            help=help_for("green_weak_green_threshold"),
        )
        weak_signal_connection_radius = st.selectbox(
            label("green_weak_signal_connection_radius"),
            options=[5, 9, 13, 17, 21, 25],
            key="green_weak_signal_connection_radius",
            help=help_for("green_weak_signal_connection_radius"),
        )
        minimum_weak_coverage_inside_hole = st.slider(
            label("green_minimum_weak_coverage_inside_hole"),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="green_minimum_weak_coverage_inside_hole",
            help=help_for("green_minimum_weak_coverage_inside_hole"),
        )
        max_hole_area_to_rescue = st.number_input(
            label("green_max_hole_area_to_rescue"),
            min_value=1,
            step=1000,
            key="green_max_hole_area_to_rescue",
            help=help_for("green_max_hole_area_to_rescue"),
        )
        fill_only_holes_with_green_evidence = st.checkbox(
            label("green_fill_only_holes_with_green_evidence"),
            key="green_fill_only_holes_with_green_evidence",
            help=help_for("green_fill_only_holes_with_green_evidence"),
        )
        st.caption(
            "Weak rescue fills only holes or weak areas with measurable green evidence."
        )

    return {
        "preset_name": preset_name,
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
        "enable_weak_green_rescue": enable_weak_green_rescue,
        "weak_green_threshold": weak_green_threshold,
        "weak_signal_connection_radius": weak_signal_connection_radius,
        "minimum_weak_coverage_inside_hole": minimum_weak_coverage_inside_hole,
        "max_hole_area_to_rescue": int(max_hole_area_to_rescue),
        "fill_only_holes_with_green_evidence": fill_only_holes_with_green_evidence,
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

    third_row = st.columns(3)
    third_row[0].metric(
        "Rescued weak green area px",
        f"{summary_metrics['rescued_weak_green_area_px']}",
    )
    third_row[1].metric(
        "Number of rescued holes",
        f"{summary_metrics['rescued_hole_count']}",
    )
    third_row[2].metric(
        "Mean weak coverage of rescued holes",
        f"{summary_metrics['mean_weak_coverage_of_rescued_holes']:.4f}",
    )


def show_green_image_grid(rgb_image, green_results: dict) -> None:
    """Render green cytoskeleton analysis image outputs."""

    row1 = st.columns(2)
    with row1[0]:
        st.image(rgb_image, caption="Original Image", use_container_width=True)
        st.download_button(
            "Download Original Image PNG",
            data=image_to_png_bytes(rgb_image),
            file_name="original_image.png",
            mime="image/png",
            key="download_green_original_image",
        )
    with row1[1]:
        st.image(
            green_results["green_channel_display"],
            caption="Green Channel",
            use_container_width=True,
        )
        st.download_button(
            "Download Green Channel PNG",
            data=image_to_png_bytes(green_results["green_channel_display"]),
            file_name="green_channel.png",
            mime="image/png",
            key="download_green_channel",
        )

    row2 = st.columns(2)
    with row2[0]:
        st.image(
            green_results["green_enhanced"],
            caption="Enhanced Green Channel",
            use_container_width=True,
        )
        st.download_button(
            "Download Enhanced Green PNG",
            data=image_to_png_bytes(green_results["green_enhanced"]),
            file_name="enhanced_green_channel.png",
            mime="image/png",
            key="download_green_enhanced",
        )
    with row2[1]:
        st.image(
            green_results["cytoskeleton_fiber_mask"],
            caption="Cytoskeleton Fiber Mask",
            use_container_width=True,
        )
        st.download_button(
            "Download Cytoskeleton Fiber Mask PNG",
            data=image_to_png_bytes(green_results["cytoskeleton_fiber_mask"]),
            file_name="green_cytoskeleton_fiber_mask.png",
            mime="image/png",
            key="download_green_fiber_mask",
        )

    st.image(
        green_results["estimated_cell_area_mask"],
        caption="Estimated Cell-covered Area Mask",
        use_container_width=True,
    )
    st.download_button(
        "Download Estimated Cell Area Mask PNG",
        data=image_to_png_bytes(green_results["estimated_cell_area_mask"]),
        file_name="estimated_cell_area_mask.png",
        mime="image/png",
        key="download_green_estimated_area_mask",
    )

    row3 = st.columns(3)
    with row3[0]:
        st.image(
            green_results["overlay_fibers"],
            caption="Overlay: Cytoskeleton Fibers",
            use_container_width=True,
        )
        st.download_button(
            "Download Fiber Overlay PNG",
            data=image_to_png_bytes(green_results["overlay_fibers"]),
            file_name="green_cytoskeleton_fibers_overlay.png",
            mime="image/png",
            key="download_green_fiber_overlay",
        )
    with row3[1]:
        st.image(
            green_results["overlay_weak_rescue"],
            caption="Overlay: Weak Green Rescued Area",
            use_container_width=True,
        )
        st.download_button(
            "Download Weak Rescue Overlay PNG",
            data=image_to_png_bytes(green_results["overlay_weak_rescue"]),
            file_name="green_weak_rescue_overlay.png",
            mime="image/png",
            key="download_green_weak_rescue_overlay",
        )
    with row3[2]:
        st.image(
            green_results["overlay_estimated_area"],
            caption="Overlay: Estimated Cell-covered Areas",
            use_container_width=True,
        )
        st.download_button(
            "Download Estimated Area Overlay PNG",
            data=image_to_png_bytes(green_results["overlay_estimated_area"]),
            file_name="estimated_cell_area_overlay.png",
            mime="image/png",
            key="download_green_estimated_area_overlay",
        )


def show_green_downloads(green_df: pd.DataFrame) -> None:
    """Expose green cytoskeleton CSV downloads."""

    st.subheader("Download Green Cytoskeleton Results")
    st.download_button(
        label="Download green cytoskeleton statistics CSV",
        data=dataframe_to_csv_bytes(green_df),
        file_name="green_cytoskeleton_statistics.csv",
        mime="text/csv",
        disabled=green_df.empty,
    )


def main() -> None:
    """Run the Streamlit app."""

    initialize_session_state(st.session_state)

    st.title("Microscopy Fluorescence Image Analyzer")
    st.caption(
        "Prototype for traditional image-processing analysis of fluorescence microscopy images."
    )

    render_profile_controls()

    analysis_mode = st.segmented_control(
        label("analysis_module"),
        options=["Blue Channel Nuclei Detection", "Green Cytoskeleton Area"],
        selection_mode="single",
        key="analysis_mode",
    )
    if analysis_mode is None:
        analysis_mode = "Blue Channel Nuclei Detection"

    st.sidebar.header(label("analysis_parameters"))
    config = None
    green_params = None
    if analysis_mode == "Blue Channel Nuclei Detection":
        render_parameter_guide("blue")
        config = build_sidebar_config()
    else:
        render_parameter_guide("green")
        green_params = build_green_parameter_panel()

    uploaded_file = st.file_uploader(
        "Upload a fluorescence microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        if analysis_mode == "Blue Channel Nuclei Detection":
            st.info("Upload a PNG, JPG, JPEG, TIF, or TIFF image to start blue nuclei analysis.")
        else:
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

    if analysis_mode == "Blue Channel Nuclei Detection":
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

        show_downloads(stats_df)
    else:
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
                enable_weak_green_rescue=green_params["enable_weak_green_rescue"],
                weak_green_threshold=green_params["weak_green_threshold"],
                weak_signal_connection_radius=green_params["weak_signal_connection_radius"],
                minimum_weak_coverage_inside_hole=green_params[
                    "minimum_weak_coverage_inside_hole"
                ],
                max_hole_area_to_rescue=green_params["max_hole_area_to_rescue"],
                fill_only_holes_with_green_evidence=green_params[
                    "fill_only_holes_with_green_evidence"
                ],
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

        show_green_downloads(green_df)


if __name__ == "__main__":
    main()
