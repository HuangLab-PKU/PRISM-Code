# Changelog

This document records all notable changes to the PRISM project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- `prism.image_process` subpackage and `scripts/image_process.py`. Image processing (focal stacking, illumination correction, registration, stitching, AIRLOCALIZE) has been split out into the sibling package `spatial_img_core` (`Huanglab/spatial_img_core/`). PRISM now starts from stitched images only.
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
