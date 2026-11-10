"""Compatibility CLI; all readout logic lives in prism.readout.pipeline."""
from prism.readout.pipeline import cli

if __name__ == "__main__":
    raise SystemExit(cli())
