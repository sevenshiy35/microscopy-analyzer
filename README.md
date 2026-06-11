# Microscopy Fluorescence Image Analyzer

展示型 Python/Streamlit 软件原型，用于分析生物显微镜拍摄的荧光细胞图像。当前实现蓝色荧光细胞核区域的自动识别、标注和定量统计，并新增绿色荧光 cytoskeleton / actin-like fibers 区域分析。不使用深度学习。

## Features

- Upload microscopy images in `png`, `jpg`, `jpeg`, `tif`, or `tiff` format.
- Normalize 16-bit TIFF and other non-8-bit images to 8-bit for display and thresholding.
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
- Estimate green cytoskeleton-covered cell area with traditional image processing:
  - Green channel extraction
  - Optional large Gaussian background subtraction
  - Gaussian blur denoising
  - Otsu or manual thresholding
  - Morphological open and close
  - Optional dilation before cell area filling
  - OpenCV flood-fill based enclosed-region filling
  - Connected component filtering and optional edge exclusion
- Quantify each estimated green cell-covered region:
  - Area
  - Bounding box
  - Centroid
  - Mean, max, and integrated green intensity
  - Cytoskeleton pixel area
  - Cytoskeleton coverage ratio
  - Edge region flag
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

## Notes for Future Extension

The UI is kept in `app.py`, while the image-processing logic is isolated in `image_analysis.py`.

The current channel utilities are designed so future modules can add:

- Red channel protein fluorescence intensity analysis
- Multi-channel colocalization statistics
- Batch image processing
- Export of annotated result folders into `sample_outputs/`
