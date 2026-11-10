"""Compatibility CLI; numerical implementation is owned by prism.readout.pipeline."""
from prism.readout.pipeline import cli, load_config, main

__all__ = ['cli', 'load_config', 'main']

if __name__ == '__main__':
    raise SystemExit(cli())
