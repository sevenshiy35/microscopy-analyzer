"""Parameter labels, defaults, presets, and JSON profile helpers."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


LANGUAGE_OPTIONS = ["English", "中文", "Bilingual / 双语"]
DEFAULT_LANGUAGE = "Bilingual / 双语"


BLUE_DEFAULTS = {
    "blue_threshold_mode": "Otsu",
    "blue_manual_threshold": 60,
    "blue_gaussian_kernel": 5,
    "blue_morph_close_kernel": 7,
    "blue_minimum_area": 1000,
    "blue_exclude_edge_nuclei": True,
    "blue_edge_margin": 5,
}


GREEN_DEFAULTS = {
    "green_analysis_target": "Estimated cell-covered area",
    "green_preset": "Balanced",
    "green_threshold_mode": "Manual",
    "green_manual_threshold": 20,
    "green_gaussian_kernel": 3,
    "green_background_subtraction": True,
    "green_background_blur_kernel": 51,
    "green_morph_open_kernel": 0,
    "green_fiber_close_kernel": 5,
    "green_area_dilation_kernel": 13,
    "green_area_close_kernel": 35,
    "green_fill_holes": False,
    "green_area_smoothing_kernel": 9,
    "green_minimum_estimated_area": 15000,
    "green_exclude_edge_regions": False,
    "green_edge_margin": 5,
    "green_weak_green_rescue": True,
    "green_weak_green_threshold": 15,
    "green_weak_signal_connection_radius": 13,
    "green_minimum_weak_coverage_inside_hole": 0.10,
    "green_max_hole_area_to_rescue": 50000,
    "green_fill_only_holes_with_green_evidence": True,
}


CYTOSKELETON_AREA_PRESETS = {
    "Conservative": {
        "green_threshold_mode": "Manual",
        "green_manual_threshold": 25,
        "green_gaussian_kernel": 3,
        "green_background_subtraction": True,
        "green_background_blur_kernel": 51,
        "green_morph_open_kernel": 0,
        "green_fiber_close_kernel": 5,
        "green_area_dilation_kernel": 9,
        "green_area_close_kernel": 25,
        "green_fill_holes": False,
        "green_area_smoothing_kernel": 9,
        "green_minimum_estimated_area": 15000,
        "green_exclude_edge_regions": False,
    },
    "Balanced": {
        "green_threshold_mode": "Manual",
        "green_manual_threshold": 20,
        "green_gaussian_kernel": 3,
        "green_background_subtraction": True,
        "green_background_blur_kernel": 51,
        "green_morph_open_kernel": 0,
        "green_fiber_close_kernel": 5,
        "green_area_dilation_kernel": 13,
        "green_area_close_kernel": 35,
        "green_fill_holes": False,
        "green_area_smoothing_kernel": 9,
        "green_minimum_estimated_area": 15000,
        "green_exclude_edge_regions": False,
    },
    "Sensitive": {
        "green_threshold_mode": "Manual",
        "green_manual_threshold": 18,
        "green_gaussian_kernel": 3,
        "green_background_subtraction": True,
        "green_background_blur_kernel": 51,
        "green_morph_open_kernel": 0,
        "green_fiber_close_kernel": 5,
        "green_area_dilation_kernel": 13,
        "green_area_close_kernel": 35,
        "green_fill_holes": False,
        "green_area_smoothing_kernel": 15,
        "green_minimum_estimated_area": 15000,
        "green_exclude_edge_regions": False,
    },
}


LABELS = {
    "language": ("Language", "语言"),
    "analysis_parameters": ("Analysis Parameters", "分析参数"),
    "analysis_module": ("Analysis module", "分析模块"),
    "parameter_profile": ("Current Parameter Profile", "当前参数配置"),
    "profile_name": ("Profile name", "参数配置名称"),
    "download_parameters": ("Download current parameters", "下载当前参数"),
    "load_parameters": ("Load parameters from JSON", "从 JSON 导入参数"),
    "blue_guide": ("Blue Nuclei Detection Guide", "蓝色细胞核检测参数说明"),
    "green_guide": ("Green Cytoskeleton Area Guide", "绿色细胞骨架面积参数说明"),
    "blue_threshold_mode": ("Blue channel threshold mode", "蓝色通道阈值模式"),
    "blue_manual_threshold": ("Manual threshold slider", "手动阈值滑块"),
    "blue_gaussian_kernel": ("Gaussian blur kernel", "高斯模糊核"),
    "blue_morph_close_kernel": ("Morph close kernel size", "形态学闭运算核大小"),
    "blue_minimum_area": ("Minimum nucleus area", "最小细胞核面积"),
    "blue_exclude_edge_nuclei": ("Exclude edge nuclei", "排除边缘细胞核"),
    "blue_edge_margin": ("Edge margin", "边缘距离"),
    "green_preset": ("Cytoskeleton area preset", "细胞骨架面积预设"),
    "green_analysis_target": ("Analysis target", "分析目标"),
    "green_threshold_mode": ("Green threshold mode", "绿色阈值模式"),
    "green_manual_threshold": ("Manual green threshold", "手动绿色阈值"),
    "green_gaussian_kernel": ("Green Gaussian blur kernel", "绿色高斯模糊核"),
    "green_background_subtraction": ("Background subtraction", "背景扣除"),
    "green_background_blur_kernel": ("Background blur kernel", "背景模糊核"),
    "green_morph_open_kernel": ("Morph open kernel size", "形态学开运算核大小"),
    "green_fiber_close_kernel": ("Fiber close kernel size", "纤维闭运算核大小"),
    "green_area_dilation_kernel": ("Area estimation dilation kernel", "面积估算膨胀核"),
    "green_area_close_kernel": ("Area estimation close kernel", "面积估算闭运算核"),
    "green_fill_holes": ("Fill holes", "填充空洞"),
    "green_area_smoothing_kernel": ("Area smoothing kernel", "面积平滑核"),
    "green_minimum_estimated_area": ("Minimum estimated area", "最小估算面积"),
    "green_exclude_edge_regions": ("Exclude edge regions", "排除边缘区域"),
    "green_edge_margin": ("Green edge margin", "绿色边缘距离"),
    "green_weak_green_rescue": ("Enable weak green rescue", "启用弱绿色信号救回"),
    "green_weak_green_threshold": ("Weak green threshold", "弱绿色阈值"),
    "green_weak_signal_connection_radius": ("Weak signal connection radius", "弱信号连接半径"),
    "green_minimum_weak_coverage_inside_hole": (
        "Minimum weak coverage inside hole",
        "空洞内最小弱信号覆盖率",
    ),
    "green_max_hole_area_to_rescue": ("Max hole area to rescue", "可救回最大空洞面积"),
    "green_fill_only_holes_with_green_evidence": (
        "Fill only holes with green evidence",
        "仅填充有绿色证据的空洞",
    ),
}


HELP = {
    "blue_threshold_mode": (
        "Controls how blue pixels are separated from background. Otsu is automatic; Manual uses the slider. Use Manual when Otsu misses weak nuclei or includes background.",
        "控制如何把蓝色像素从背景中分离。Otsu 为自动阈值；Manual 使用手动滑块。当 Otsu 漏检弱细胞核或包含背景时可改用 Manual。",
    ),
    "blue_manual_threshold": (
        "Higher values keep only brighter blue nuclei; lower values include weaker blue signal and may add noise.",
        "数值越高越只保留亮蓝色细胞核；数值越低越容易包含弱蓝色信号，也可能增加噪声。",
    ),
    "blue_gaussian_kernel": (
        "Smooths noise before thresholding. Larger kernels reduce speckles but can blur small nuclei; smaller kernels preserve detail.",
        "阈值前平滑噪声。核越大越能减少斑点，但可能模糊小细胞核；核越小越保留细节。",
    ),
    "blue_morph_close_kernel": (
        "Fills small holes and connects gaps inside nuclei. Larger values close more gaps; smaller values preserve shape.",
        "填补细胞核内部小洞并连接断裂。数值越大连接越强；数值越小越保留原形。",
    ),
    "blue_minimum_area": (
        "Removes small detected objects. Increase it to suppress noise; decrease it when real nuclei are small.",
        "过滤小目标。调大可去除噪声；真实细胞核较小时应调小。",
    ),
    "blue_exclude_edge_nuclei": (
        "Removes nuclei touching or near image borders. Turn on when border nuclei are incomplete; turn off to count all visible regions.",
        "排除接触或靠近图像边缘的细胞核。边缘核不完整时建议开启；需要统计所有可见区域时关闭。",
    ),
    "blue_edge_margin": (
        "Distance from the image border used for edge exclusion. Larger margins exclude more border objects.",
        "边缘排除使用的距离。数值越大，排除的边缘目标越多。",
    ),
    "green_analysis_target": (
        "Chooses whether the display emphasizes fluorescent fibers or estimated cell-covered area. Region IDs always summarize larger estimated regions.",
        "选择显示重点是荧光纤维还是估算细胞覆盖面积。Region_ID 始终表示较大的估算区域。",
    ),
    "green_threshold_mode": (
        "Controls strong green fiber segmentation. Manual is predictable; Otsu adapts automatically but may be unstable with uneven fluorescence.",
        "控制强绿色纤维分割。Manual 更可控；Otsu 自动适应，但在荧光不均时可能不稳定。",
    ),
    "green_manual_threshold": (
        "Higher values keep only bright fibers; lower values include dim cytoskeleton but may include background.",
        "数值越高越只保留亮纤维；数值越低越能包含淡绿色骨架，但可能包含背景。",
    ),
    "green_gaussian_kernel": (
        "Denoises green signal. Smaller values preserve thin fibers; larger values smooth noise but may erase fine structures.",
        "绿色信号去噪。小核保留细纤维；大核更平滑但可能抹掉细结构。",
    ),
    "green_background_subtraction": (
        "Reduces uneven background using a large blur. Enable for haze or gradients; disable if it removes real weak signal.",
        "用大尺度模糊扣除不均匀背景。有光晕或梯度时开启；如果削弱真实弱信号可关闭。",
    ),
    "green_background_blur_kernel": (
        "Controls background scale. Larger values model broader haze; smaller values remove more local variation and can over-correct.",
        "控制背景尺度。数值越大越适合宽范围光晕；数值越小越可能过度扣除局部信号。",
    ),
    "green_morph_open_kernel": (
        "Optional noise removal for fibers. Increase to remove speckles; keep 0 when thin fibers are being lost.",
        "可选的纤维噪声去除。调大可去斑点；细纤维被删掉时保持 0。",
    ),
    "green_fiber_close_kernel": (
        "Connects small breaks in strong fibers. Larger values bridge more gaps; smaller values avoid merging separate fibers.",
        "连接强纤维中的小断裂。数值越大连接越多；数值越小更避免误合并。",
    ),
    "green_area_dilation_kernel": (
        "Thickens fibers before area estimation. Larger values create broader covered areas; smaller values stay closer to fibers.",
        "面积估算前加粗纤维。数值越大估算区域越宽；数值越小越贴近原纤维。",
    ),
    "green_area_close_kernel": (
        "Connects cytoskeleton-supported networks into larger regions. Larger values merge gaps; smaller values keep regions separated.",
        "把骨架网络连接成较大区域。数值越大越容易合并；数值越小保留分离。",
    ),
    "green_fill_holes": (
        "Fills all enclosed holes. Usually off for cytoskeleton because internal gaps may be biological; enable only when holes are clearly artifacts.",
        "填充所有封闭空洞。细胞骨架图像通常关闭，因为内部间隙可能有生物意义；只有确定为空洞伪影时开启。",
    ),
    "green_area_smoothing_kernel": (
        "Smooths estimated area boundaries. Larger values produce smoother regions but may remove narrow structures.",
        "平滑估算面积边界。数值越大边界越平滑，但可能移除狭窄结构。",
    ),
    "green_minimum_estimated_area": (
        "Filters small estimated regions. Increase to remove fragments; decrease when real cells or regions are small.",
        "过滤小的估算区域。调大可去碎片；真实区域较小时调小。",
    ),
    "green_exclude_edge_regions": (
        "Removes estimated regions near image borders. Enable when edge regions are incomplete.",
        "排除靠近图像边缘的估算区域。边缘区域不完整时开启。",
    ),
    "green_edge_margin": (
        "Distance from border for green edge exclusion. Larger values exclude more peripheral regions.",
        "绿色区域边缘排除距离。数值越大，排除更多外围区域。",
    ),
    "green_weak_green_rescue": (
        "Adds dim green evidence only when supported by strong nearby fibers or qualifying holes. Turn off for very noisy images.",
        "仅在有邻近强纤维或合格空洞证据时加入淡绿色信号。图像噪声很强时可关闭。",
    ),
    "green_weak_green_threshold": (
        "Threshold for dim green candidates. Lower values rescue more weak signal but can add background; higher values are stricter.",
        "淡绿色候选阈值。调低可救回更多弱信号但可能带入背景；调高更严格。",
    ),
    "green_weak_signal_connection_radius": (
        "Distance used to connect weak signal to strong fibers. Larger values rescue farther weak pixels; smaller values are more conservative.",
        "弱信号连接到强纤维的距离。调大可救回更远弱像素；调小更保守。",
    ),
    "green_minimum_weak_coverage_inside_hole": (
        "Minimum fraction of weak green pixels required inside a hole before it is filled. Higher values fill fewer holes.",
        "空洞被填充前所需的最小弱绿色像素比例。数值越高，填充空洞越少。",
    ),
    "green_max_hole_area_to_rescue": (
        "Upper area limit for selective hole rescue. Larger values allow larger holes; smaller values protect large real background gaps.",
        "选择性空洞救回的面积上限。调大允许更大空洞；调小可保护大面积真实背景间隙。",
    ),
    "green_fill_only_holes_with_green_evidence": (
        "If enabled, holes are filled only when weak green evidence is present. Disable only when using global Fill holes intentionally.",
        "开启后只填充有弱绿色证据的空洞。只有明确需要全局填洞时才关闭。",
    ),
}


GUIDES = {
    "blue": [
        (
            "Threshold controls which blue pixels are treated as nuclei.",
            "阈值决定哪些蓝色像素被视为细胞核。",
        ),
        ("Gaussian blur reduces noise.", "高斯模糊用于降低噪声。"),
        (
            "Morph close fills small holes inside nuclei.",
            "形态学闭运算用于填补细胞核内部小洞。",
        ),
        ("Minimum area removes small noise.", "最小面积用于去除小噪声区域。"),
        (
            "Edge exclusion removes incomplete nuclei touching image borders.",
            "边缘排除用于去掉接触图像边界的不完整细胞核。",
        ),
    ],
    "green": [
        (
            "Strong green fibers represent clear cytoskeleton signal.",
            "强绿色纤维表示明确的细胞骨架信号。",
        ),
        (
            "Estimated cell-covered area is not the same as green pixel area.",
            "估算细胞覆盖面积不等于绿色像素面积。",
        ),
        (
            "Dilation and close estimate larger cytoskeleton-supported regions.",
            "膨胀和闭运算用于估算由骨架支持的更大区域。",
        ),
        (
            "Fill holes should usually be off for cytoskeleton images because internal gaps can be biologically meaningful.",
            "细胞骨架图像中通常不建议全局填洞，因为内部间隙可能有生物意义。",
        ),
        (
            "Weak green rescue can recover dim green regions supported by nearby stronger cytoskeleton.",
            "弱绿色救回可以恢复由邻近强骨架支持的淡绿色区域。",
        ),
    ],
}


def text(key: str, language: str) -> str:
    """Return a label in the selected language."""

    english, chinese = LABELS[key]
    if language == "English":
        return english
    if language == "中文":
        return chinese
    return f"{english} / {chinese}"


def help_text(key: str, language: str) -> str:
    """Return help text in the selected language."""

    english, chinese = HELP[key]
    if language == "English":
        return english
    if language == "中文":
        return chinese
    return f"{english}\n\n{chinese}"


def guide_lines(module: str, language: str) -> list[str]:
    """Return bilingual guide lines for the requested module."""

    lines = []
    for english, chinese in GUIDES[module]:
        if language == "English":
            lines.append(f"- {english}")
        elif language == "中文":
            lines.append(f"- {chinese}")
        else:
            lines.append(f"- {english}\n  {chinese}")
    return lines


def all_defaults() -> dict[str, Any]:
    """Return a fresh copy of all UI defaults."""

    defaults: dict[str, Any] = {
        "language": DEFAULT_LANGUAGE,
        "profile_name": "balanced_green_cytoskeleton",
        "analysis_mode": "Blue Channel Nuclei Detection",
    }
    defaults.update(BLUE_DEFAULTS)
    defaults.update(GREEN_DEFAULTS)
    return deepcopy(defaults)


def initialize_session_state(session_state: Any) -> None:
    """Populate missing Streamlit session keys without overwriting user edits."""

    for key, value in all_defaults().items():
        session_state.setdefault(key, value)


def apply_green_preset(session_state: Any, preset_name: str) -> None:
    """Apply a green cytoskeleton preset to session state."""

    preset = CYTOSKELETON_AREA_PRESETS.get(preset_name)
    if not preset:
        return
    for key, value in preset.items():
        session_state[key] = value
    session_state["green_preset"] = preset_name


def sanitize_profile_name(profile_name: str) -> str:
    """Return a filesystem-friendly profile name for downloads."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile_name.strip())
    return cleaned or "microscopy_profile"


def build_profile(session_state: Any) -> dict[str, Any]:
    """Build a JSON-serializable parameter profile from session state."""

    profile_name = session_state.get("profile_name", "balanced_green_cytoskeleton")
    return {
        "profile_name": profile_name,
        "language": session_state.get("language", DEFAULT_LANGUAGE),
        "blue_nuclei": {
            "threshold_mode": session_state.get("blue_threshold_mode"),
            "manual_threshold": session_state.get("blue_manual_threshold"),
            "gaussian_kernel": session_state.get("blue_gaussian_kernel"),
            "morph_close_kernel": session_state.get("blue_morph_close_kernel"),
            "minimum_area": session_state.get("blue_minimum_area"),
            "exclude_edge_nuclei": session_state.get("blue_exclude_edge_nuclei"),
            "edge_margin": session_state.get("blue_edge_margin"),
        },
        "green_cytoskeleton": {
            "analysis_target": session_state.get("green_analysis_target"),
            "preset": session_state.get("green_preset"),
            "threshold_mode": session_state.get("green_threshold_mode"),
            "manual_threshold": session_state.get("green_manual_threshold"),
            "gaussian_kernel": session_state.get("green_gaussian_kernel"),
            "background_subtraction": session_state.get("green_background_subtraction"),
            "background_blur_kernel": session_state.get("green_background_blur_kernel"),
            "morph_open_kernel": session_state.get("green_morph_open_kernel"),
            "fiber_close_kernel": session_state.get("green_fiber_close_kernel"),
            "area_dilation_kernel": session_state.get("green_area_dilation_kernel"),
            "area_close_kernel": session_state.get("green_area_close_kernel"),
            "fill_holes": session_state.get("green_fill_holes"),
            "area_smoothing_kernel": session_state.get("green_area_smoothing_kernel"),
            "minimum_estimated_area": session_state.get("green_minimum_estimated_area"),
            "exclude_edge_regions": session_state.get("green_exclude_edge_regions"),
            "edge_margin": session_state.get("green_edge_margin"),
            "weak_green_rescue": session_state.get("green_weak_green_rescue"),
            "weak_green_threshold": session_state.get("green_weak_green_threshold"),
            "weak_signal_connection_radius": session_state.get(
                "green_weak_signal_connection_radius"
            ),
            "minimum_weak_coverage_inside_hole": session_state.get(
                "green_minimum_weak_coverage_inside_hole"
            ),
            "max_hole_area_to_rescue": session_state.get("green_max_hole_area_to_rescue"),
            "fill_only_holes_with_green_evidence": session_state.get(
                "green_fill_only_holes_with_green_evidence"
            ),
        },
    }


def profile_to_json_bytes(profile: dict[str, Any]) -> bytes:
    """Serialize a profile dictionary to pretty JSON bytes."""

    return json.dumps(profile, indent=2, ensure_ascii=False).encode("utf-8")


def load_profile_bytes(raw_bytes: bytes) -> dict[str, Any]:
    """Parse parameter profile JSON bytes."""

    data = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("profile must be an object")
    return data


def apply_profile(session_state: Any, profile: dict[str, Any]) -> None:
    """Apply a partial JSON parameter profile to Streamlit session state."""

    defaults = all_defaults()
    if "profile_name" in profile:
        session_state["profile_name"] = profile["profile_name"] or defaults["profile_name"]
    if "language" in profile and profile["language"] in LANGUAGE_OPTIONS:
        session_state["language"] = profile["language"]

    blue = profile.get("blue_nuclei", {})
    if isinstance(blue, dict):
        mapping = {
            "threshold_mode": "blue_threshold_mode",
            "manual_threshold": "blue_manual_threshold",
            "gaussian_kernel": "blue_gaussian_kernel",
            "morph_close_kernel": "blue_morph_close_kernel",
            "minimum_area": "blue_minimum_area",
            "exclude_edge_nuclei": "blue_exclude_edge_nuclei",
            "edge_margin": "blue_edge_margin",
        }
        for source_key, state_key in mapping.items():
            if source_key in blue:
                session_state[state_key] = blue[source_key]

    green = profile.get("green_cytoskeleton", {})
    if isinstance(green, dict):
        mapping = {
            "analysis_target": "green_analysis_target",
            "preset": "green_preset",
            "threshold_mode": "green_threshold_mode",
            "manual_threshold": "green_manual_threshold",
            "gaussian_kernel": "green_gaussian_kernel",
            "background_subtraction": "green_background_subtraction",
            "background_blur_kernel": "green_background_blur_kernel",
            "morph_open_kernel": "green_morph_open_kernel",
            "fiber_close_kernel": "green_fiber_close_kernel",
            "area_dilation_kernel": "green_area_dilation_kernel",
            "area_close_kernel": "green_area_close_kernel",
            "fill_holes": "green_fill_holes",
            "area_smoothing_kernel": "green_area_smoothing_kernel",
            "minimum_estimated_area": "green_minimum_estimated_area",
            "exclude_edge_regions": "green_exclude_edge_regions",
            "edge_margin": "green_edge_margin",
            "weak_green_rescue": "green_weak_green_rescue",
            "weak_green_threshold": "green_weak_green_threshold",
            "weak_signal_connection_radius": "green_weak_signal_connection_radius",
            "minimum_weak_coverage_inside_hole": (
                "green_minimum_weak_coverage_inside_hole"
            ),
            "max_hole_area_to_rescue": "green_max_hole_area_to_rescue",
            "fill_only_holes_with_green_evidence": (
                "green_fill_only_holes_with_green_evidence"
            ),
        }
        for source_key, state_key in mapping.items():
            if source_key in green:
                session_state[state_key] = green[source_key]

