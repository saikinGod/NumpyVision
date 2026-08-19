# NumpyVision

A personal learning project focused on understanding the core mathematics of image processing from scratch. Instead of relying on heavy libraries like OpenCV, this project uses **NumPy** to manipulate image arrays directly, demonstrating how concepts like matrix multiplication, convolution, and array slicing power modern computer vision.

## Project Goal

To learn and implement low-level image processing techniques using pure mathematics and vectorized operations in Python.

## Directory Structure

```text
NUMPYVISION/
│
├── .venv/                   # Virtual Environment
├── numpyVision/             # Core Package Directory
│   ├── __init__.py          # Package initializer
│   ├── core.py              # Main CoreNpVision class & inplace logic
│   ├── ops_color.py         # Color manipulation math (RGB/Grayscale)
│   ├── ops_filters.py       # Convolution matrices (Kernels)
│   └── ops_geometry.py      # Spatial operations & array slicing
│
├── main.py                  # Entry point for testing the code
├── numpyTesting.jpeg        # Sample test image
├── numpyTesting.png         # Sample test image (RGBA/Float format)
└── README.md                # Project documentation
```

## Key Features & Capabilities

### 1. Smart Architecture

- **Method Chaining:** Process images seamlessly (e.g., `img.to_grayscale().invert_colors().show()`).
- **Non-Destructive Editing:** Supports `inplace=False` to return new image instances without altering the original array, mimicking professional data science libraries like Pandas.
- **Format Handling:** Automatically converts PNG floats (0.0 - 1.0) to standard `uint8` (0-255) and safely strips the Alpha channel (RGBA to RGB) for consistent mathematical operations.

### 2. Color Operations (`ops_color.py`)

Direct manipulation of pixel values handling the Integer Overflow Trap (`uint8` clipping).

- Adjust Brightness & Contrast
- Grayscale Conversion (Luminosity mean)
- Vintage Sepia (Matrix dot product)
- Color Inversion (Negative effect)
- RGB Channel Tinting & Color Balance
- Solarize (Threshold-based inversion)

### 3. Geometry & Spatial Operations (`ops_geometry.py`)

Utilizing advanced NumPy slicing and broadcasting.

- Crop & Center Crop
- Flip (Horizontal/Vertical)
- Rotate 90°
- Add Border (Canvas padding)
- Mirror Effect (Symmetry via axis broadcasting)

### 4. Advanced Filters & Convolution (`ops_filters.py`)

Custom-built 3x3 kernel engine using vectorized processing for high-speed execution without nested loops.

- **Box Blur:** Local neighborhood averaging.
- **Edge Detection (Laplacian):** Highlighting sharp color transitions.
- **Sharpen:** Contrast enhancement on edges.
- **Emboss:** 3D shadow and highlight illusion.

## Usage Example

This is how the library is utilized within `main.py`:

```python
from numpyVision import NumpyVision

def main():
    # Load the image
    img = NumpyVision("numpyTesting.png")

    # Example 1: Chain operations and display
    img.adjust_brightness(30).apply_sepia().add_border(15, color=(0,0,0)).show("Vintage Framed")

    # Example 2: Non-destructive filtering using inplace=False
    edges_img = img.edge_detect(inplace=False)
    edges_img.show("Edge Detection Result")

if __name__ == "__main__":
    main()
```

## Core Concepts Learned

- **NumPy Array Manipulation:** Slicing, transposing, and channel isolation.
- **Type Casting & Data Loss:** Managing `uint8`, `int16`, and `float32` types to prevent overflow bugs.
- **Broadcasting:** Applying arithmetic operations across different array shapes.
- **Image Convolution:** Understanding how 3x3 kernels interact with neighboring pixels to create visual effects.