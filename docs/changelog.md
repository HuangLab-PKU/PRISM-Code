# Changelog

This document records all notable changes to the PRISM project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Automatic QC
- Readout and gene-calling QC PNG/JSON are now required pipeline artifacts for
  shared readout, PR31/PR46, VIS and file-producing generic classification.
- Reports use complete counts and explicitly bounded samples; absent pure-Cy3
  posteriors and missing fitting/entropy diagnostics remain unavailable.
- Calling-only imports obtain readout QC automatically. A QC-only backfill command
  and source-bound VIS numerical checkpoint prevent refitting just to repair a plot.
- See [automatic QC](qc.md) for outputs, denominators, limits and recovery.

### Fixed
- **The two shipped configs could not be chained.** `readout.yaml` derived intensity column
  names from image filename stems (`cyc_1_cy5`), while `gene_calling_base.yaml` keyed
  `channel_correction.channels` on `r01_ex628`, so the readout → gene-calling handoff died
  with a `KeyError` inside `apply_channel_correction`. Channel *files* and channel *names*
  are now declared separately (`channels: [{file, name}]`), letting upstream acquisition own
  the filenames while PRISM owns the column vocabulary. The legacy `channel_files:` list
  still works and keeps its file-stem naming. New `prism.readout.channels.resolve_channels`,
  with the stage contract pinned by `tests/test_shipped_configs.py`.
- **`load_gene_calling_config` silently discarded the main file's own body.** The merge was
  seeded from `config_modules` and `overrides` only, so passing a content file (one without
  a `config_modules` key) straight to the pipeline yielded `{}` — 130 lines of
  `gene_calling_base.yaml` evaporating without a warning. Layering is now modules → main
  body → overrides. A missing module file also fails fast instead of logging a warning and
  producing a partial config.
- **Colourspace visualization fabricated missing data.** `_preprocess_for_visualization`
  filled absent `color_dim_1` / `color_dim_2` / `ch3/A` / `G_layer` columns with
  `np.random`, rendering a plausible-looking figure of pure noise on a schema mismatch. It
  now raises `ValueError` naming every missing column.

### Changed
- Spotiflow's `prob_thresh` and torch `device` are read from `readout.yaml`
  (`detection_prob_thresh`, `device`) instead of being hardcoded in `scripts/readout.py`.
  `device: null` auto-selects CUDA when available and falls back to CPU, replacing the
  previous unconditional `'cuda'`.
- `scripts/segment_dapi.py --base-dir` and `readout.yaml:base_dir` no longer default to an
  internal lab NAS path; `--base-dir` is now required and the config ships a placeholder.
- `pyproject.toml` sets `testpaths = ["tests"]` so collection stops walking into
  `experiments/`.

### Added
- A shared `python -m prism.pipeline` production entry for the explicit `prism46_v1`
  profile, with package-owned readout, preprocessing, seeded GMM, annotation,
  manual-gating figures, technical checks and source-bound stage receipts.
- Complete panel CSV validation and explicit configuration replace filename-based
  target inference. Later stages reject changed predecessor recipes and outputs.
- Readout worker failures propagate; incomplete intensities are not filled with zero.
  Loading a new readout config resets omitted settings instead of inheriting a prior batch.
- Manual-gating colour-space exports now accompany the channel-space overview: one
  scatter/density pair per layer at 800 x 700 pixels, with an exact pixel-coordinate
  sidecar. These use all assigned spots rather than the overview subsample.
- **Posterior-confidence QC for gene calling.** The GMM decoder's per-spot top-1 posterior
  is now surfaced as a decode-confidence score. `ClassificationResult.to_dataframe(...,
  confidence_threshold=t)` adds an `is_confident` column; `SignalClassificationPipeline`
  writes it to `predictions.csv` and generates a confident-only ColorSpace via the new
  `ClassificationEvaluator.visualize_confident_colorspace(...)`, which drops ambiguous
  between-codeword spots for a cleaner, higher-specificity view. Configurable through
  `classification.confidence_threshold` (default `0.8`; set to `null` to disable). New
  `prism.gene_calling.confidence` module (`top1_confidence`, `confident_mask`) with unit
  tests in `tests/test_confidence.py`.

- **Stitched mosaic input (OME-TIFF / OME-Zarr).** Current `spatial_img_core` stitching
  writes one `stitched/mosaic.ome.tif` (finalized) or `stitched/mosaic.ome.zarr` (in
  progress) instead of one `cyc_1_<channel>.tif` per channel, and `scripts/readout.py` /
  `scripts/segment_dapi.py` could not open either. Both now read all three layouts. Configs
  are unchanged: a channel is still named `cyc_1_cy5.tif`, and on a mosaic that name
  selects cycle 1 / channel `cy5` inside it. When a mosaic and per-channel TIFFs are both
  present, the mosaic is read. Blocks are read lazily from the compressed tiles, so the
  whole plane is never loaded. New `prism.readout.mosaic` (a copy of
  `sprintseq.readout.mosaic`; reads the format without depending on `spatial_img_core`)
  and `prism.readout.stitched` (filename addressing on top). Needs the new `mosaic`
  extra: `pip install -e ".[mosaic]"` (`zarr`, `imagecodecs`). Tests:
  `tests/test_mosaic*.py`, `tests/test_stitched.py`, and
  `tests/test_readout_layouts.py`, which checks that readout output is identical whichever
  layout holds the pixels.

### Documentation overhaul
- **Fixed** broken doc cross-links (hyphenated filenames in `README.md`), removed dead links (`api-reference.md`, `tutorial.md`, `PRISM_gene_calling`), and corrected the pipeline flowchart — Spot Detection / Readout is now inside the PRISM stage and the diagram is drawn left-to-right.
- **Changed** all docs to be relative to the PRISM-Code repository root; rewrote `installation.md` in English with no machine-specific environments or hard-coded paths; rewrote `configuration.md` to document the real `readout.yaml` and modular `gene_calling*.yaml` configs; aligned `detailed-usage.md` / `quick-start.md` with the current scripts (`readout.py`, `gene_calling.py`, `segment_dapi.py`) and the spotiflow detector; output filenames in `data-architecture.md` now match the scripts (`position.csv`, `intensity.csv`, `mapping.csv`, `dapi_centroids.csv`). `spatial_img_core` is now described as a not-yet-public companion package (contact huanglab111@gmail.com).
- **Removed** `docs/lazy-imports-optimization.md` and `docs/memmap-optimization.md` (legacy `spot_detection` framework) and `docs/postcode-usage.md` (PoSTcode demoted to an experimental, opt-in gene-calling method).

### Removed
- `prism.image_process` subpackage and `scripts/image_process.py`. Image processing (focal stacking, illumination correction, registration, stitching, AIRLOCALIZE) has been split out into the companion package `spatial_img_core` (not yet public). PRISM now starts from stitched images only.
- `SimpleITK` dependency (only `image_process/register_sitk.py` consumed it).
- Six image-processing notebooks were moved out of `code/notebooks/` to `spatial_img_core/notebooks/`: `image_process_3D`, `density_im_generate_2D`, `density_im_generate_3D`, `cell_segment_confocal_3D`, `dapi_prediction_confocal_3D`, `convex_hull_projection_confocal_3D`.

### Changed
- `pyproject.toml` description / classifier / dependency-comment now reflect post-stitching scope.
- Root `CLAUDE.md`, `code/README.md`, and all `code/docs/*.md` updated to point to `spatial_img_core` for upstream image processing and to correct stale `code/src/` references (actual layout is flat `code/prism/`).
- `code/scripts/legacy/README.md` carries a WARNING block at the top: scripts that import the removed `prism.image_process` (multi_channel_readout*, spot_detection_pipeline, image_scan_fstack, image_process_pipeline, pipeline_3D) no longer run as-is.

### Added
- Comprehensive documentation structure in `docs/` directory
- YAML-based configuration system for multi-channel readout
- Channel transformation matrix for scaling and crosstalk elimination
- Modular spot detection functions in `src/spot_detection/`
- Refactored multi-channel readout scripts with improved maintainability
- `MultiChannelProcessor` class for encapsulating processing logic
- Configuration loader functions in `src/spot_detection/config_loader.py`
- Channel transformation utilities in `src/spot_detection/channel_transformation.py`
- Default configuration file: `configs/default_multi_channel_readout.yaml`
- Matrix validation and utility functions for channel transformation
- Delayed import mechanism to avoid dependency issues

### Changed
- Refactored `multi_channel_readout.py` into modular components
- Consolidated scaling factors and crosstalk elimination into transformation matrix
- Improved code organization and reusability
- Enhanced parameter management through YAML configuration files
- Moved spot detection functions to `src/spot_detection/multi_channel_processor.py`
- Updated documentation to English throughout the project
- Simplified main README by removing redundant Quick Start content
- Reorganized configuration guide to cover all PRISM components
- Moved batch processing files to `local/` directory

### Removed
- API Reference documentation (replaced with usage guides)
- Redundant functions moved to appropriate modules
- Separate documentation files (content integrated into changelog)
- Batch processing references from main documentation

## [2024.12] - Initial Refactoring

### Added
- `MultiChannelProcessor` class for encapsulating processing logic
- Configuration loader functions in `src/spot_detection/config_loader.py`
- Channel transformation utilities in `src/spot_detection/channel_transformation.py`
- Default configuration file: `configs/default_multi_channel_readout.yaml`

### Changed
- Moved spot detection functions to `src/spot_detection/multi_channel_processor.py`
- Implemented delayed imports to avoid dependency issues
- Updated documentation to English throughout the project

### Technical Details
- **Modularization**: Separated concerns into focused modules
- **Configuration Management**: Centralized parameter management
- **Error Handling**: Improved error messages and validation
- **Documentation**: Comprehensive guides for installation, usage, and configuration

## [2024.11] - Original Implementation

### Added
- Initial PRISM pipeline implementation
- Multi-channel image processing capabilities
- Spot detection and gene calling workflows
- Cell segmentation tools
- 2D and 3D analysis support

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024.12 | Major refactoring with modular design and YAML configuration |
| 0.9.0 | 2024.11 | Initial implementation of PRISM pipeline |

## Contributing

When making changes to this project, please update this changelog by:

1. Adding new entries under the appropriate version section
2. Following the format: `### Added/Changed/Removed/Fixed`
3. Including brief descriptions of the changes
4. Updating the version number if it's a significant change

## Notes

- This changelog focuses on user-facing changes and significant technical improvements
- Internal refactoring and code organization changes are documented here
- Breaking changes are clearly marked and include migration instructions when applicable
