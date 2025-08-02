# PRISM_Code

PRISM (**P**rofiling of **R**NA **I**n-situ through **S**ingle-round i**M**aging) is an innovative method that employs an radius vector code to distinguish a wide array of RNA transcripts in large-scale tissues with sub-micron resolution through a single staining and imaging cycle, which make it fast and free of prblems of traditional methods like shifting and alignment.

For more information, please read the article: [PRISM: Multiplexed Profiling of RNA In-Situ through Single-round of Imaging in Three-Dimensional Tissue](https://doi.org/10.1101/2024.06.29.601330).

# Code and Pipeline Preview

Code for PRISM consists of the following parts: **probe_designer**, **image_process**, **gene_calling** **cell_segmentation**, **analysis_cell_typing** and **analysis_suncellular**. Data will be processed in this order.

The pipeline can be explained as:

```mermaid
graph TD
    A["Probe Design"] --> B["Experiment"];

    subgraph "Image Data Acquisition"
        B --> C["2D Data Stack"];
        B --> E["3D Data"];
    end

    subgraph "Image Processing"
        C --> D["2D Image Process"];
        E --> F["3D Image Process"];
    end

    D --> G("Gene Calling");
    F --> G("Gene Calling");

    subgraph "Segmentation & Analysis"
        G --> H["2D Cell Segment"];
        G --> I["3D Cell Segment"];
        H --> J["Analysis"];
        I --> J["Analysis"];
    end
```

# Quick Start for PRISM decoding

## clone thie repo to your repo/app path

```bash
git clone --depth 1 https://github.com/tangmc0210/PRISM_code
```

## Environment

We need python and MATLAB as environment. For matlab, Image Process Toolbox is needed, and for python, We recommend using `Python 3.8` for installing the required packages. The packages listed in the `requirements.txt` file are necessary.

```shell
pip install -r requirements.txt
```

And MATLAB engine should be installed, according to your local MATLAB version, you can follow [official guideline](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

Additionally, **pip** and **setuptools** need to be at the correct version. For MATLAB R2021b, run:

```
pip install --upgrade pip setuptools==57.5.0
```

Then, install the MATLAB Engine API for Python with:

```
python -m pip install matlabengine==9.11.21
```

## Start from stitched raw images (relatively simple)

1. Download our data from zenodo [MouseEmbryo Data](https://zenodo.org/records/13219763). And put them like the [Data architecture](#data-architecture).

2. Read signals using `multi_channel_readout.py` like described in step [image process2d-step3](#-Step-3:-Multi_channel_readout).

3. Call genes (spots classification) using ` PRISM_gene_calling_EMBRYO_30.ipynb` provided in [MouseEmbryo Data](https://zenodo.org/uploads/12750725?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjNmMGQzNDhjLWYwOTItNDkzMy04MzEyLWNjNzVkNWY2YTc4NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDJhYjhkMGRiYmM3NjE0MzJmNTVmZWE2MTNhZTcwZSJ9.T-2r0mRpqdI4cOm0Dl_vYfcpqjObMqVgL4eFWlmL-6eCMdbfVdRzpHZj9Ld9OdRXksJ9dMft5ui09AyJJqaadQ) at the position `20221219_PRISM_E13.5_2_3_Three_processed/readout/ PRISM_gene_calling_EMBRYO_30.ipynb`. Example output is located in `20221219_PRISM_E13.5_2_3_Three_processed/readout/Mannual_publication`.

Then you get the result as `mapped_genes.csv` which include the RNA spots and their positions, from images of each fluoroscence channel (`20221219_PRISM_E13.5_2_3_Three_processed/stitched/*.tif`).

## Start from unstitched raw images

1. Download our data from NetDisk [HCC_raw_images](https://disk.pku.edu.cn/link/AA382E67AE9779469C97814C27892A43DF) (you may need to download the Tiles in the cyc_1_zip for many batched because of the bandwidth limit). UseAnd put them like the [Data architecture](#data-architecture).

2. Image process

   - Stack the z series images using `scan_fstack.py`.
   - Image process using `image_process_after_stack.py`.
   - Read signals using `multi_channel_readout.py`.

3. Call genes (spots classification) using `PRISM_gene_calling_2d_HCC.ipynb` provided in [HCC_Data](https://zenodo.org/uploads/12750711?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0ZTViMzkxLTAyZmEtNGRjNy1hYzhmLTU1OTYyNGVhYzA3YyIsImRhdGEiOnt9LCJyYW5kb20iOiI5YzJhNWY1ZDZmMjlhMjI3MDJjOTA4Mzk0OTliZDQ5NyJ9.VUDVHe4llxiAC9lyRDTWL6g1YAp4fjFObJ3yfe_Bau0oFkA2C0yFLEiTFCabVUA1qHQy9NP5jPv8n6Dxxiw7rg) at the position `20230523_HCC_PRISM_probe_refined_processed/readout/PRISM_gene_calling_2d_HCC.ipynb`. Example output is located in `20230523_HCC_PRISM_probe_refined_processed/readout/mannual_publication`.

Then you get the result as `mapped_genes.csv` which include the RNA spots and their positions, from raw images of each fluoroscence channel (`cyc_*_*/*.tif`).

# Data and Architecture

## How to get our data

Stitched raw images are provided in zenodo.org, download from it based on your need.

> 1. [MouseEmbryo](https://zenodo.org/records/13219763)
> 2. [HCC](https://zenodo.org/records/13208941)
> 3. [MouseBrain3D](https://zenodo.org/records/12673246)
> 4. [Cell typing and Analysis](https://zenodo.org/records/12755414)

We also provided **HCC2D** unstitched raw images in [PkuNetdisk](https://disk.pku.edu.cn/link/AA83FADBB90EB14BAE8E9DE5889E94AFF9) (you may need to download the Tiles in the cyc_1_zip for many batched because of the bandwidth limit). Download them and decompress the images using the script in the folder.

**For more raw data, contact us: huanglab111@gmail.com.**

## Data architecture

Raw data base directory and processed data output directory can be whatever place you need. But its subdirectory should be like this:

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

```shell
Output root
├─RUN_ID1_processed        # automate create
│  ├─focal_stacked         # automate create
│  ├─background_corrected  # automate create, will be deleted after image process
│  ├─resized               # automate create, will be deleted after image process
│  ├─registered            # automate create
│  ├─stitched              # automate create
│  ├─segmented             # automate create
│  └─readout               # automate create
├─RUN_ID2_processed        # automate create
├─...
└─RUN_IDN_processed        # automate create
```

Your raw data should be in folder RUN_ID.

Important results of each step are stored in corresponding directory under RUN_ID_processed. In this example, steps' dir are defined as:

```shell
# In image process
dest_dir = BASE_DIR / f'{RUN_ID}_processed' # processed data
aif_dir = dest_dir / 'focal_stacked'        # scan_fstack.py
sdc_dir = dest_dir / 'background_corrected' # image_process_after_stack.py
rsz_dir = dest_dir / 'resized'              # image_process_after_stack.py
rgs_dir = dest_dir / 'registered'           # image_process_after_stack.py
stc_dir = dest_dir / 'stitched'             # image_process_after_stack.py

# In following analysis
src_dir = BASE_DIR / f'{RUN_ID}_processed'  # processed data
stc_dir = src_dir / 'stitched'              # image_process_after_stack.py
read_dir = src_dir / 'readout'              # multi_channel_readout.py
seg_dir = src_dir / 'segmented'             # segment2D.py or segment3D.py or expression_matrix.py
visual_dir = src_dir / 'visualization'      # folder for figures...
```

# Tutorial
**\*Remind!** Lots of paths or directories need editing in files mentioned below.\*

We provide jupyter notebooks to show how to use PRISM to perform the analysis. The notebooks are located in the `notebooks` folder.

## Probe Design

This step is not always necessary because you can design the probe with specific binding sites, barcodes and corresponding fluorophore probes manually or contact us for help. But if you want to design the probe easily or in bulk, see: [probe_designer](https://github.com/tangmc0210/probe_designer).


---

## Image Process 2D

Step 1 and Step 2 is used to generate a complete image for each channel used in experiment. If you have other methods to per form this image process, store the name of each channel's image as `cyc_1_channel.tif`

### Step 1: Scan_fstack

Edit the directory in python file `scan_fstack.py` and run the code:

```shell
python scripts/scan_fstack.py Raw_data_root
```

**Remark**: This step is aimed to process raw images captured in small field and multi channel. you can use it to process your own experiment data. We have provided a preprocessed example data for Step 2 and pipeline after, which is located at `./dataset/processed/_example_dataset_processed`. You can change the RUN_ID in each script to `_example_dataset` and continue the following steps.

### Step 2: Image_process

Edit the directory in python file `image_process/image_process_after_stack.py` as the same, and run the code:

```shell
python scripts/image_proess_after_stack.py
```

**Remark**: This step includes register the subimages, correct the background and stitch them to a whole image. Results will be stitched n (n is the number of channels you use) big images in `stc_dir`, which will be used in next part.

> ### 3D reconstructure of 2D images
>
> **TODO**
>
> If your images are captured as mentioned in [Data Architecture](#Data-Architecture) above and you want to restore the z stack infomation(even if only 10um), change the parameters file path in `pipeline_3D.py` and run:
>
> ```shell
> python pipeline_3D.py
> ```
>
> to read the intensity from raw images.
>
> **Remark**: This pipeline includes 2D process as cycle shift and global position of each tile is needed from 2D pipeline. After that, airlocalize is performed to extract spots in 3d (z stack number as the depth). Remember to change the parameters file path and adjust the parameters for your own data before you run the code.

## Spot detection 2D
### Feature based Spot Detection
Edit the directory in python file `scripts/multi_channel_readout.py` as the same, and run the code:

```shell
python scripts/multi_channel_readout.py
```

**Remark**: This step requires stitched big images generated in the previous step. Signal spots and their intensity can be extract using `scripts/multi_channel_readout.py`. It will generate two csv files named `tmp/intensity_all.csv` `intensity_all_deduplicated.csv` in the directory `RUN_IDx_processed/readout/` and copy the .py file to the readout path, too.

### Deep Learning Based Spot Detection (Recommended)

This updated workflow uses a StarDist deep learning model to detect spots from multi-channel images, providing higher accuracy and avoiding the issue of duplicate detections for spots that appear in multiple channels.

#### Step 1: Prepare Training Data

The most critical step is to create high-quality training data. This involves annotating your images to tell the model what a "spot" looks like.

1.  **Understand the Data Structure**:
    -   Your training images should be placed in `data/training/images/`. These must be **multi-channel TIFF files**, with the shape `(channels, height, width)`.
    -   Your corresponding masks go in `data/training/masks/`. These must be **single-channel TIFF files** where each individual spot is "painted" with a unique integer ID (1, 2, 3, ...). The background must be 0.

2.  **Annotate Your Images**:
    -   We highly recommend using **Fiji/ImageJ** with the **Labkit** plugin for this task.
    -   Load your multi-channel image into Fiji, then open it in Labkit.
    -   On a single label layer, carefully paint over every unique spot you see across all channels. Labkit will automatically assign a unique ID to each disconnected spot you paint.
    -   For spots that are very close, ensure their masks do not touch. Use the eraser tool or the Watershed method to create a 1-pixel separation.
    -   Export the final annotation from Labkit using `Save > Export Labeling as Tiff...` and save it as a `Unsigned 16-bit` TIFF.
    -   For more detailed instructions, see the guide in `src/spot_detection/README.md`.

#### Step 2: Train the Model

Once you have prepared at least 10-20 annotated image/mask pairs, you can train the model.

-   Run the training script from your terminal:
    ```bash
    python scripts/train_spot_detector.py --use-gpu
    ```
-   This will use the data in `data/training/`, train a new model, and save it to the `models/` directory. You can adjust training parameters like epochs and patch size directly in the command line. Run `python scripts/train_spot_detector.py --help` for more options.

#### Step 3: Run Inference

After the model is trained, you can use it to detect and quantify spots in new, unseen images.

-   Run the inference script, providing the path to your images and where to save the output:
    ```bash
    python scripts/multi_channel_readout_dp.py \
        --input-dir /path/to/your/stitched/images \
        --output-csv /path/to/your/readout/results.csv \
        --channel-files cyc_1_cy5.tif cyc_1_TxRed.tif cyc_1_cy3.tif cyc_1_FAM.tif \
        --channels cy5 TxRed cy3 FAM
    ```
-   This script will:
    1.  Load your trained StarDist model from the `models/` directory.
    2.  Combine your single-channel images into a multi-channel stack.
    3.  Detect all unique spots.
    4.  For each spot, fit a 2D Gaussian to measure its integrated intensity and local background in every channel.
    5.  Save the results to a `.csv` file, with columns like `Y`, `X`, `cy5_intensity`, `cy5_background`, etc.

## Spot detection 3D
If your images are captured by confocal, lightsheet or any other 3D microscopy and you have a registered and stitched grayscale 3d image of each channel in tiff format.

We recommand you using [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE) in matlab to perform spots extraction because its well designed user interface for adjustment of proper parameters. Open MATLAB and run `AIRLOCALIZE.m` in `src/Image_process/lib/AIRLOCALIZE-MATLAB`. The input files should locate at `path_to_runid/RUN_ID_processed/stitched` and the output path should be `path_to_runid/RUN_ID_processed/readout/tmp`.

> Alternatively, spot extraction can be performed using airlocalize.py with proper parameters (set at `Image_process\lib\AIRLOCALIZE\parameters.yaml`).
>
> ```shell
> python image_process/lib/AIRLOCALIZE/airlocalize.py
> ```
>
> **Remarks**:
>
> - The default image axis is 'XYZ' in airlozalize, if you need other axis order like 'ZXY', please use np.transpose() in previous step or modify the 'self.retrieve_img()' function in `Image_process/lib/AIRLOCALIZE/airlocalize_data.py`.
>
> - AIRLOCALIZE includes predetection signal spots on a feature image(like DoG or LoG or Gaussian smoothed) and fit the accurate location and intensity of spots on original image. Important parameters includes:
>
> ```
> # scaling
> scale: scale the image to drop too low or too high spots.
> scaleMaxRatio: the final absolute upperbound of scaled image, don't set it 1 to prevent excess of np.uint16 and cause errors.
>
> # predetection
> featureExtract: feature image method, LoG is thought best while it takes relatively high calculation resources. DoG with properly set filterLo and filterHi can be seen as a similar replacement.
> ThreshLevel: list, different channel can be set different. The higher, the less points detected.
> maxSpots: predetected spots number, got by strip the first 'maxSpots' spots sorted by intensity from high to low.
>
> # Gaussian fit
> Psfsigma: the sigma of your gaussian like signal points, determined by your real data.
> ```

After that, intensity decoding and gene calling can be performed using `gene_calling\readout_gene_calling_3d.ipynb`.

---

## Gene Calling

In this part, we recommand using `gene_calling/gene_calling_GMM.ipynb` when you have `readout/intensity.csv` because spots distribution in color space may differ between tissue types or cameras, but for a quick start, you can also use `gene_calling/gene_calling_GMM.py` by editting the directory in python file `gene_calling/gene_calling_GMM.py` and run the code:

```shell
python scripts/gene_calling_GMM.py
```

The result should be at `read_dir/mapped_genes.csv` as default.

**Remark**:

- Gene calling for PRISM is performed by a Gaussian Mixture, mannual select by masks and evaluation of the confidence of each spot. It's expected to run on a GUI because some steps need human knowledge of the experiments like how the chemical environment or FRET would affect the fluorophores.

- You can also use `gene_calling/PRISM_gene_calling_GMM.ipynb` for customization or use `gene_calling/gene_calling_manual.ipynb` to set the threshold for each gene manually.

- 3D gene calling in our article was performed in `gene_calling\PRISM3D_intensity_readout_and_gene_calling.ipynb`.

For more detail, see [PRISM_gene_calling](https://github.com/tangmc0210/PRISM_gene_calling).

## Cell Segmentation

### Dapi centroids

Edit the directory in python file `cell_segmentation/segment2D.py` or `cell_segmentation/segment3D.py` and run:

```shell
python scripts/segment2D.py
```

or

```shell
python scripts/segment3D.py
```

This code will segment cell nucleus according to DAPI channel. A csv file containing the coordinate of nucleus centroid will be generated in`seg_dir` as `centroids_all.csv`.

### Expression matrix

Edit the directory in python file `gene_calling/expression_matrix.py`, and run:

```shell
python scripts/expression_matrix2D.py
```

or

```shell
python scripts/expression_matrix3D.py
```

the expression matrix will be generated in `seg_dir` as `expression_matrix.csv`

**Remarks**:

- `Segmentation3D.py` needs stardist environment as it use trained network to predict the shape and centroid of nucleus in 3D. For more information, see: [stardist](https://github.com/stardist/stardist).
- Our strategy to generate expression matris in general assign rna to its nearest centroid of cell nucleus (predicted by dapi) so it requires `dapi_centroids.csv` of cell nucleus and `mapped_genes.csv` generated in previous steps. If you have other strategies which performed better in your data, you can replace this step with it.

---