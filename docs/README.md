# PRISM Documentation

Welcome to the PRISM documentation! This directory contains comprehensive guides for using the PRISM (**P**rofiling of **R**NA **I**n-situ through **S**ingle-round i**M**aging) method and tools.

## Documentation Structure

### Getting Started
- **[Quick Start](quick-start.md)** - Get up and running with PRISM quickly
- **[Installation Guide](installation.md)** - Complete installation instructions including MATLAB setup

### User Guides
- **[Detailed Usage Guide](detailed-usage.md)** - Comprehensive workflow documentation
- **[Configuration Guide](configuration.md)** - Parameter configuration and customization

### Technical Reference
- **[API Reference](api-reference.md)** - Complete API documentation for developers

## Quick Navigation

### For New Users
1. Start with [Quick Start](quick-start.md) to understand the basic workflow
2. Follow [Installation Guide](installation.md) to set up your environment
3. Use [Detailed Usage Guide](detailed-usage.md) for step-by-step instructions

### For Experienced Users
1. Check [Configuration Guide](configuration.md) for parameter tuning
2. Refer to [API Reference](api-reference.md) for advanced usage
3. Use [Detailed Usage Guide](detailed-usage.md) for specific workflow steps

### For Developers
1. Review [API Reference](api-reference.md) for complete function documentation
2. Check [Configuration Guide](configuration.md) for configuration file structure
3. Use [Installation Guide](installation.md) for development environment setup

## Key Features

### PRISM Post-Stitching Pipeline
- **Spot Detection**: Detect RNA spots from stitched images using feature-based or deep learning methods
- **Gene Calling**: Identify genes using Gaussian Mixture Models (GMM / codebook-GMM / PoSTcode)
- **Cell Segmentation**: Segment cells via DAPI and generate expression matrices

Upstream (probe design, image acquisition / stitching) is handled by the sibling [`probe_designer`](https://github.com/tangmc0210/probe_designer) and `spatial_img_core` (sibling repo) packages.

### Refactored Multi-Channel Readout
- **Modular Design**: Clean, maintainable code structure
- **Configuration Management**: YAML-based parameter management
- **Transformation Matrix**: Unified channel correction system

## Data Sources

### Sample Data
- [MouseEmbryo Data](https://zenodo.org/records/13219763)
- [HCC Data](https://zenodo.org/records/13208941)
- [MouseBrain3D Data](https://zenodo.org/records/12673246)
- [Cell Typing and Analysis](https://zenodo.org/records/12755414)

### Raw Data
- [HCC Raw Images](https://disk.pku.edu.cn/link/AA382E67AE9779469C97814C27892A43DF)

## External Resources

- **Probe Design**: [probe_designer](https://github.com/tangmc0210/probe_designer)
- **3D Segmentation**: [StarDist](https://github.com/stardist/stardist)
- **3D Spot Detection**: [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE)
- **Gene Calling**: [PRISM_gene_calling](https://github.com/tangmc0210/PRISM_gene_calling)

## Support

For questions or support, contact us at: **huanglab111@gmail.com**

## Citation

If you use PRISM in your research, please cite:
- [BioArxiv 2024.6.29](https://doi.org/10.1101/2024.06.29.601330)
