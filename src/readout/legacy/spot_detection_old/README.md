# Data Preparation for Spot Detection Training

This guide explains how to prepare your data for training the spot detection model. The model is based on StarDist, which requires image and mask pairs for training.

## 1. Data Structure

Organize your training data into the following directory structure:

```
data/
└── training/
    ├── images/
    │   ├── sample_01.tif
    │   ├── sample_02.tif
    │   └── ...
    └── masks/
        ├── sample_01.tif
        ├── sample_02.tif
        └── ...
```

-   `images/`: This folder should contain your raw **multi-channel** 2D TIFF images. Each file should be a stack where each slice represents a different channel (e.g., a 4-channel image would have the shape `(4, height, width)`).
-   `masks/`: This folder should contain the corresponding single-channel 2D ground truth segmentation masks. Each mask should be a label image where every unique spot (regardless of which channel it appears in) is represented by a unique integer ID.

## 2. Creating Masks

The quality of your training data, especially the masks, is crucial for the model's performance. You can use an annotation tool like [napari](https://napari.org/) or [Fiji/ImageJ](https://imagej.net/Fiji) to create the masks.

### Using napari for Annotation

1.  **Install napari**:
    ```bash
    pip install "napari[all]"
    ```

2.  **Load your multi-channel image**:
    Open napari and drag-and-drop one of your multi-channel TIFF images into the viewer. It should open as a stack.

3.  **Create a new labels layer**:
    Click the "New labels layer" button in the layer list.

4.  **Annotate your spots**:
    -   Go through the different channels of your image using the slider.
    -   On the single labels layer, annotate all unique spots you see across all channels. If a spot appears in multiple channels, you only need to annotate it once.
    -   Select the "paint" tool and carefully paint over each spot. Each spot should be a distinct object in the mask.
    -   Ensure that each spot has a unique integer label.

5.  **Save the mask**:
    -   Select the labels layer you created.
    -   Go to `File > Save Selected Layer(s)...`.
    -   Save the file as a TIFF image (`.tif`) in the `masks` directory. Make sure the filename matches the corresponding image in the `images` directory. The mask should be saved as a `uint16` or `uint32` image.

### Annotation Tips

-   **Be consistent**: Annotate spots with a consistent size and shape.
-   **Cover all examples**: Include examples of spots that appear in single channels and spots that appear in multiple channels.
-   **Background**: The background in the mask should have a value of 0.

## 3. Training and Validation Split

The training script will automatically split your data into training and validation sets. A typical split is 90% for training and 10% for validation. You should have at least 10-20 images with high-quality annotations to start with. The more data, the better the model will perform.