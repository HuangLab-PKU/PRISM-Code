"""
Setup script for PRISM code package.

Install in development mode:
    cd code
    pip install -e .

This allows importing from src modules directly:
    from src.gene_calling.pipeline import SignalClassificationPipeline
    from src.readout.spot_detection import get_spot_coordinates
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="prism",
    version="0.1.0",
    description="PRISM: Probe-based Imaging for Spatial Transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Huang Lab",
    packages=find_packages(exclude=["notebooks", "projects", "local", "legacy"]),
    python_requires=">=3.8",
    install_requires=requirements,
    # Include non-Python files if needed
    include_package_data=True,
    # Entry points for command-line scripts (optional)
    # entry_points={
    #     "console_scripts": [
    #         "prism-readout=scripts.readout:main",
    #         "prism-gene-calling=scripts.gene_calling:main",
    #     ],
    # },
)
