"""Resolution of stitched-image files to canonical channel names.

Upstream stitching names the TIFFs on disk (``cyc_1_cy5.tif``); PRISM names the
intensity columns it emits (``r01_ex628``, encoding imaging round and excitation
wavelength).  Holding those apart means the gene-calling stage can rely on a
stable column vocabulary without anyone renaming acquisition output, and
without a separate header-rewriting pass between the two stages.

Configuration accepts either form::

    channels:                       # preferred: filename and column name
      - {file: cyc_1_cy5.tif, name: r01_ex628}
      - {file: cyc_1_cy3.tif, name: r01_ex531}

    channel_files:                  # legacy: column name is the file stem
      - cyc_1_cy5.tif
"""

from pathlib import Path
from typing import Any


def resolve_channels(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Resolve a readout config's channel declaration.

    Args:
        config: Parsed ``readout.yaml`` contents.  Must contain either
            ``channels`` (list of ``{file, name}`` mappings, ``name`` optional)
            or the legacy ``channel_files`` (list of filenames).

    Returns:
        ``(filename, channel_name)`` pairs in declaration order.  The order
        defines the column order of the emitted intensity table.

    Raises:
        KeyError: If neither key is present, or a ``channels`` entry has no
            ``file``.
        ValueError: If the declaration is empty or names are not unique.
    """
    if "channels" in config:
        entries = config["channels"]
    elif "channel_files" in config:
        entries = [{"file": filename} for filename in config["channel_files"]]
    else:
        raise KeyError(
            "Readout config must declare 'channels' (preferred) or "
            "'channel_files' (legacy)"
        )

    if not entries:
        raise ValueError("Readout config must declare at least one channel")

    resolved: list[tuple[str, str]] = []
    for entry in entries:
        if "file" not in entry:
            raise KeyError(f"Channel entry is missing 'file': {entry}")
        filename = str(entry["file"])
        resolved.append((filename, str(entry.get("name") or Path(filename).stem)))

    names = [name for _, name in resolved]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Channel names must be unique; duplicated: {duplicates}")

    return resolved
