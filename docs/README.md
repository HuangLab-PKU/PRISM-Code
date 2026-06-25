# PRISM Documentation

Welcome to the PRISM documentation! This directory contains comprehensive guides for using the PRISM (**P**rofiling of **R**NA **I**n-situ through **S**ingle-round i**M**aging) post-stitching pipeline.

## Documentation Structure

### Getting Started
- **[Quick Start](quick-start.md)** - Get up and running with PRISM quickly
- **[Installation Guide](installation.md)** - Complete installation instructions, optional extras, and GPU notes

### User Guides
- **[Detailed Usage Guide](detailed-usage.md)** - Comprehensive, step-by-step workflow documentation
- **[Configuration Guide](configuration.md)** - Parameter configuration for readout, gene calling, and segmentation
- **[Data Architecture](data-architecture.md)** - Data structure and directory layout requirements

### Reference
- **[Changelog](changelog.md)** - Project change history

## Quick Navigation

### For New Users
1. Start with [Quick Start](quick-start.md) to understand the basic workflow
2. Follow [Installation Guide](installation.md) to set up your environment
3. Use [Detailed Usage Guide](detailed-usage.md) for step-by-step instructions

### For Experienced Users
1. Check [Configuration Guide](configuration.md) for parameter tuning
2. Use [Detailed Usage Guide](detailed-usage.md) for specific workflow steps

## Key Features

### PRISM Post-Stitching Pipeline
- **Spot Detection / Readout**: Detect RNA spots from stitched images (default: spotiflow; fallback: tophat-based traditional methods) and read out per-channel intensities
- **Gene Calling**: Assign genes via Gaussian Mixture Models (GMM / codebook-GMM; PoSTcode available as an experimental option)
- **Cell Segmentation**: Segment nuclei from DAPI and build cell-by-gene expression matrices

Upstream steps (probe design, image acquisition / stitching) are handled by the companion [`probe_designer`](https://github.com/tangmc0210/probe_designer) and `spatial_img_core` packages. `spatial_img_core` is **not yet public** — request access at **huanglab111@gmail.com**.

## Data Sources

See [Data Sources in the main README](../README.md#data-sources) for the Zenodo sample datasets and raw-image downloads.

## External Resources

- **Probe Design**: [probe_designer](https://github.com/tangmc0210/probe_designer)
- **3D Segmentation**: [StarDist](https://github.com/stardist/stardist)
- **3D Spot Detection**: [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE) (invoked through `spatial_img_core`)

## Support

For questions or support, contact us at: **huanglab111@gmail.com**

## Citation

If you use PRISM in your research, please cite:
- [Nature Biotechnology (2025)](https://doi.org/10.1038/s41587-025-02883-7)
- [bioRxiv (2024)](https://doi.org/10.1101/2024.06.29.601330)
