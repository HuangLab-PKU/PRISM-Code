"""Run the shared calling stages from an explicit run configuration."""
import sys
from prism.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main(["--stage", "calling", *sys.argv[1:]]))
