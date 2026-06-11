# Microscopy Fluorescence Image Analyzer

展示型 Python/Streamlit 软件原型，用于分析生物显微镜拍摄的荧光细胞图像。当前实现蓝色荧光细胞核区域的自动识别、标注和定量统计，并新增绿色荧光 cytoskeleton / actin-like fibers 区域分析。不使用深度学习。

## Features

- Upload microscopy images in `png`, `jpg`, `jpeg`, `tif`, or `tiff` format.
- Normalize 16-bit TIFF and other non-8-bit images to 8-bit for display and thresholding.
- Bilingual parameter interface:
  - English
  - 中文
  - Bilingual / 双语
- Parameter Guide panels explain the major blue nuclei and green cytoskeleton controls.
- Save and load parameter profiles as JSON files for reproducible analysis sessions.
- Detect blue fluorescence nuclei with traditional image processing:
  - Blue channel extraction
  - Gaussian blur denoising
  - Otsu or manual thresholding
  - Morphological close and open
  - Connected component detection
  - Area filtering and optional edge nucleus exclusion
- Display:
  - Original Image
  - Blue Channel
  - Binary Mask
  - Cleaned Mask
  - Overlay with Detected Nuclei
- Quantify each nucleus:
  - Area
  - Bounding box
  - Centroid
  - Mean, max, and integrated blue intensity
  - Edge nucleus flag
- Estimate green cytoskeleton-covered cell area with two traditional image-processing layers:
  - Cytoskeleton fiber detection: detects fluorescent cytoskeleton pixels.
  - Estimated cell-covered area estimation: thickens, connects, fills, and filters cytoskeleton networks to approximate larger covered regions.
  - Weak Green Signal Rescue: uses a lower weak-green threshold only near strong fibers or in enclosed holes with measurable green evidence.
  - Green channel extraction
  - Optional large Gaussian background subtraction
  - Small Gaussian blur denoising to preserve fine fibers
  - Otsu or manual thresholding
  - Optional light morphological open for fiber noise removal
  - Small fiber close for broken cytoskeleton lines
  - Larger dilation and close for cell-covered area estimation
  - OpenCV flood-fill based enclosed-region filling
  - Connected component filtering and optional edge exclusion
- Quantify each estimated green cell-covered region:
  - Area
  - Bounding box
  - Centroid
  - Mean, max, and integrated green intensity
  - Cytoskeleton fiber pixel area inside the estimated region
  - Cytoskeleton coverage ratio
  - Edge region flag
- Green cytoskeleton area presets:
  - Conservative: cleaner boundaries and less over-estimation.
  - Balanced: recommended default for cytoskeleton-supported area estimation.
  - Sensitive: includes weaker green signals but may include more background.
  - Custom: manually tune all parameters.
- Download:
  - CSV statistics
  - Blue nuclei overlay and mask PNG
  - Green cytoskeleton mask PNG
  - Estimated cell area mask PNG
  - Green cytoskeleton overlay PNG
  - Estimated area overlay PNG

## Project Structure

```text
microscopy_analyzer/
  app.py
  image_analysis.py
  requirements.txt
  README.md
  sample_outputs/
```

## Installation

Create and activate a Python virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the project directory:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Parameter Profiles

The sidebar includes a `Current Parameter Profile` section.

To save settings:

1. Enter a profile name.
2. Click `Download current parameters`.
3. The app downloads `microscopy_parameters_PROFILE_NAME.json`.

To restore settings:

1. Use `Load parameters from JSON`.
2. Upload a previously saved JSON profile.
3. The app restores available blue nuclei and green cytoskeleton parameters.

Missing fields fall back to defaults, so older profile files remain usable.

## Presets

The Green Cytoskeleton Area module includes four preset choices:

- `Conservative`: reduces over-estimation and keeps cleaner boundaries.
- `Balanced`: recommended default for cytoskeleton-supported area estimation.
- `Sensitive`: keeps more weak green signal but may include more background.
- `Custom`: lets users manually tune all parameters.

## Notes for Future Extension

The UI is kept in `app.py`, while the image-processing logic is isolated in `image_analysis.py`.

The Green Cytoskeleton Area workflow is a display-oriented traditional image
processing estimate. It is not a deep learning single-cell segmentation model:
fiber pixels and estimated cell-covered areas are reported as separate layers.
The `Balanced` preset is the default. `Fill holes` is disabled by default because
empty spaces inside cytoskeleton networks can be biologically meaningful rather
than segmentation holes.
Weak Green Signal Rescue is enabled by default and is intentionally selective:
weak pixels must be near strong cytoskeleton support, and hole rescue requires
weak-green coverage evidence under the configured hole-size limit.

The current channel utilities are designed so future modules can add:

- Red channel protein fluorescence intensity analysis
- Multi-channel colocalization statistics
- Batch image processing
- Export of annotated result folders into `sample_outputs/`
