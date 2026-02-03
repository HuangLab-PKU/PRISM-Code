# Detailed Usage Guide

## Data Architecture

### Raw Data Structure

```shell
Raw data root
├─RUN_ID1
│  └─cyc1
│     ├─C001-T0001-cy3-Z000.tif
│     ├─C001-T0001-cy3-Z001.tif
│     ├─...
│     ├─C001-T0004-FAM-Z006.tif
│     ├─...
│     └─C001-T0108-TxRed-Z008.tif
├─RUN_ID2
├─...
└─RUN_IDN
```

### Output Data Structure

```shell
Output root
├─RUN_ID1_processed        # auto-created
│  ├─focal_stacked         # auto-created
│  ├─background_corrected  # auto-created, deleted after image processing
│  ├─resized               # auto-created, deleted after image processing
│  ├─registered            # auto-created
│  ├─stitched              # auto-created
│  ├─segmented             # auto-created
│  └─readout               # auto-created
├─RUN_ID2_processed        # auto-created
├─...
└─RUN_IDN_processed        # auto-created
```

### Directory Descriptions

```shell
# In image processing
dest_dir = BASE_DIR / f'{RUN_ID}_processed' # processed data
aif_dir = dest_dir / 'focal_stacked'        # scan_fstack.py
sdc_dir = dest_dir / 'background_corrected' # image_process_after_stack.py
rsz_dir = dest_dir / 'resized'              # image_process_after_stack.py
rgs_dir = dest_dir / 'registered'           # image_process_after_stack.py
stc_dir = dest_dir / 'stitched'             # image_process_after_stack.py

# In subsequent analysis
src_dir = BASE_DIR / f'{RUN_ID}_processed'  # processed data
stc_dir = src_dir / 'stitched'              # image_process_after_stack.py
read_dir = src_dir / 'readout'              # multi_channel_readout.py
seg_dir = src_dir / 'segmented'             # segment2D.py or segment3D.py or expression_matrix.py
visual_dir = src_dir / 'visualization'      # folder for figures...
```

## Complete PRISM Workflow

### 1. Probe Design

This step is not always necessary because you can design probes with specific binding sites, barcodes and corresponding fluorophore probes manually or contact us for help. However, if you want to design probes easily or in bulk, see: [probe_designer](https://github.com/tangmc0210/probe_designer).

### 2. Image Processing

#### 2D Image Processing

Steps 1 and 2 are used to generate a complete image for each channel used in the experiment. If you have other methods to perform this image processing, store the name of each channel's image as `cyc_1_channel.tif`.

##### Step 1: Scan_fstack

Edit the directory in the Python file `scan_fstack.py` and run the code:

```bash
python scripts/scan_fstack.py Raw_data_root
```

**Remark**: This step processes raw images captured in small fields and multiple channels. You can use it to process your own experimental data. We have provided a preprocessed example dataset for Step 2 and subsequent pipeline steps, located at `./dataset/processed/_example_dataset_processed`. You can change the RUN_ID in each script to `_example_dataset` and continue with the following steps.

##### Step 2: Image_process

Edit the directory in the Python file `image_process/image_process_after_stack.py` accordingly, and run the code:

```bash
python scripts/image_process_after_stack.py
```

**Remark**: This step includes registering the subimages, correcting the background, and stitching them into a whole image. Results will be stitched into n big images (where n is the number of channels you use) in `stc_dir`, which will be used in the next part.

##### 3D Reconstruction of 2D Images

If your images are captured as mentioned in [Data Architecture](data_architecture.md) above and you want to restore the z-stack information (even if only 10μm), change the parameters file path in `pipeline_3D.py` and run:

```bash
python pipeline_3D.py
```

to read the intensity from raw images.

**Remark**: This pipeline includes 2D processing as cycle shift and global position of each tile is needed from the 2D pipeline. After that, airlocalize is performed to extract spots in 3D (z-stack number as the depth). Remember to change the parameters file path and adjust the parameters for your own data before running the code.

### 3. Spot Detection

#### 2D Spot Detection

##### Feature-based Spot Detection

Edit the directory in the Python file `scripts/multi_channel_readout.py` accordingly, and run the code:

```bash
python scripts/multi_channel_readout.py
```

**Remark**: This step requires stitched big images generated in the previous step. Signal spots and their intensity can be extracted using `scripts/multi_channel_readout.py`. It will generate two CSV files named `tmp/intensity_all.csv` and `intensity_all_deduplicated.csv` in the directory `RUN_IDx_processed/readout/` and copy the .py file to the readout path as well.

##### Deep Learning Based Spot Detection (Recommended)

This updated workflow uses a StarDist deep learning model to detect spots from multi-channel images, providing higher accuracy and avoiding the issue of duplicate detections for spots that appear in multiple channels.

###### Step 1: Prepare Training Data

The most critical step is to create high-quality training data. This involves annotating your images to teach the model what a "spot" looks like.

1. **Understand the Data Structure**:
   - Your training images should be placed in `data/training/images/`. These must be **multi-channel TIFF files**, with the shape `(channels, height, width)`.
   - Your corresponding masks go in `data/training/masks/`. These must be **single-channel TIFF files** where each individual spot is "painted" with a unique integer ID (1, 2, 3, ...). The background must be 0.

2. **Annotate Your Images**:
   - We highly recommend using **Fiji/ImageJ** with the **Labkit** plugin for this task.
   - Load your multi-channel image into Fiji, then open it in Labkit.
   - On a single label layer, carefully paint over every unique spot you see across all channels. Labkit will automatically assign a unique ID to each disconnected spot you paint.
   - For spots that are very close, ensure their masks do not touch. Use the eraser tool or the Watershed method to create a 1-pixel separation.
   - Export the final annotation from Labkit using `Save > Export Labeling as Tiff...` and save it as an `Unsigned 16-bit` TIFF.
   - For more detailed instructions, see the guide in `src/spot_detection/README.md`.

###### Step 2: Train the Model

Once you have prepared at least 10-20 annotated image/mask pairs, you can train the model.

- Run the training script from your terminal:
  ```bash
  python scripts/train_spot_detector.py --use-gpu
  ```
- This will use the data in `data/training/`, train a new model, and save it to the `models/` directory. You can adjust training parameters like epochs and patch size directly in the command line. Run `python scripts/train_spot_detector.py --help` for more options.

###### Step 3: Run Inference

After the model is trained, you can use it to detect and quantify spots in new, unseen images.

- Run the inference script, providing the path to your images and where to save the output:
  ```bash
  python scripts/multi_channel_readout_dp.py \
      --input-dir /path/to/your/stitched/images \
      --output-csv /path/to/your/readout/results.csv \
      --channel-files cyc_1_cy5.tif cyc_1_TxRed.tif cyc_1_cy3.tif cyc_1_FAM.tif \
      --channels cy5 TxRed cy3 FAM
  ```
- This script will:
  1. Load your trained StarDist model from the `models/` directory.
  2. Combine your single-channel images into a multi-channel stack.
  3. Detect all unique spots.
  4. For each spot, fit a 2D Gaussian to measure its integrated intensity and local background in every channel.
  5. Save the results to a `.csv` file, with columns like `Y`, `X`, `cy5_intensity`, `cy5_background`, etc.

#### 3D Spot Detection

If your images are captured by confocal, light-sheet or any other 3D microscopy and you have a registered and stitched grayscale 3D image of each channel in TIFF format:

We recommend using [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE) in MATLAB to perform spot extraction because of its well-designed user interface for adjusting proper parameters. Open MATLAB and run `AIRLOCALIZE.m` in `src/Image_process/lib/AIRLOCALIZE-MATLAB`. The input files should be located at `path_to_runid/RUN_ID_processed/stitched` and the output path should be `path_to_runid/RUN_ID_processed/readout/tmp`.

> Alternatively, spot extraction can be performed using airlocalize.py with proper parameters (set at `Image_process\lib\AIRLOCALIZE\parameters.yaml`).
>
> ```bash
> python image_process/lib/AIRLOCALIZE/airlocalize.py
> ```

After that, intensity decoding and gene calling can be performed using `gene_calling\readout_gene_calling_3d.ipynb`.

### 4. Gene Calling

In this part, we recommend using `gene_calling/gene_calling_GMM.ipynb` when you have `readout/intensity.csv` because spots distribution in color space may differ between tissue types or cameras. For a quick start, you can also use `gene_calling/gene_calling_GMM.py` by editing the directory in the Python file `gene_calling/gene_calling_GMM.py` and running the code:

```bash
python scripts/gene_calling_GMM.py
```

The result should be at `read_dir/mapped_genes.csv` by default.

**Remark**:

- Gene calling for PRISM is performed by a Gaussian Mixture Model, manual selection by masks, and evaluation of the confidence of each spot. It's expected to run on a GUI because some steps need human knowledge of the experiments, such as how the chemical environment or FRET would affect the fluorophores.

- You can also use `gene_calling/PRISM_gene_calling_GMM.ipynb` for customization or use `gene_calling/gene_calling_manual.ipynb` to set the threshold for each gene manually.

- 3D gene calling in our article was performed in `gene_calling\PRISM3D_intensity_readout_and_gene_calling.ipynb`.

For more details, see [PRISM_gene_calling](https://github.com/tangmc0210/PRISM_gene_calling).

### 5. Cell Segmentation

#### DAPI Centroids

Edit the directory in the Python file `cell_segmentation/segment2D.py` or `cell_segmentation/segment3D.py` and run:

```bash
python scripts/segment2D.py
```

or

```bash
python scripts/segment3D.py
```

This code will segment cell nuclei according to the DAPI channel. A CSV file containing the coordinates of nucleus centroids will be generated in `seg_dir` as `centroids_all.csv`.

#### Expression Matrix

Edit the directory in the Python file `gene_calling/expression_matrix.py`, and run:

```bash
python scripts/expression_matrix2D.py
```

or

```bash
python scripts/expression_matrix3D.py
```

The expression matrix will be generated in `seg_dir` as `expression_matrix.csv`.

**Remarks**:

- `Segmentation3D.py` needs a StarDist environment as it uses a trained network to predict the shape and centroid of nuclei in 3D. For more information, see: [StarDist](https://github.com/stardist/stardist).
- Our strategy to generate expression matrices generally assigns RNA to its nearest centroid of cell nucleus (predicted by DAPI), so it requires `dapi_centroids.csv` of cell nuclei and `mapped_genes.csv` generated in previous steps. If you have other strategies that perform better on your data, you can replace this step with them.

## Refactored Multi-Channel Readout Script

### Overview

The refactored multi-channel readout script modularizes the original single-file code, improving maintainability and reusability. Main improvements include:

1. **Modular Design**: Moved spot detection related functions to `src/spot_detection` module
2. **Configuration Management**: Use YAML configuration files to manage all parameters
3. **Transformation Matrix**: Integrated scaling factors and crosstalk elimination into unified transformation matrix
4. **Class Encapsulation**: Use `MultiChannelProcessor` class to encapsulate processing logic

### Usage Methods

#### 1. Single File Processing

```python
from scripts.multi_channel_readout_refactored import MultiChannelProcessor

# Use default configuration
processor = MultiChannelProcessor(run_id='20250717_FFPE_OSCC')
intensity = processor.process_single_run()

# Use custom configuration
processor = MultiChannelProcessor(
    config_path='path/to/custom_config.yaml',
    run_id='20250717_FFPE_OSCC',
    prism_panel='PRISM30'
)
intensity = processor.process_single_run()
```

#### 2. Command Line Usage

```bash
# Single file processing
python scripts/multi_channel_readout_refactored.py
```

## Important Notes

**⚠️ Important:** Many paths or directories need editing in files mentioned below.

We provide Jupyter notebooks to demonstrate how to use PRISM for analysis. The notebooks are located in the `notebooks` folder.

## Additional Resources

- For probe design: [probe_designer](https://github.com/tangmc0210/probe_designer)
- For 3D segmentation: [StarDist](https://github.com/stardist/stardist)
- For 3D spot detection: [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE)

For questions or support, contact us at: **huanglab111@gmail.com**