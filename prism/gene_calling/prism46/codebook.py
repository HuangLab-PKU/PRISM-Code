import itertools
GLAYER = 3
COLOR_GRADE = 5
PURE_G_PROBE = 46
PROBE_GRADES: dict[int, tuple[int, int, int, int]] = {1: (4, 0, 0, 0), 2: (3, 1, 0, 0), 3: (2, 2, 0, 0), 4: (1, 3, 0, 0), 5: (0, 4, 0, 0), 6: (0, 3, 0, 1), 7: (0, 2, 0, 2), 8: (0, 1, 0, 3), 9: (0, 0, 0, 4), 10: (1, 0, 0, 3), 11: (2, 0, 0, 2), 12: (3, 0, 0, 1), 13: (2, 1, 0, 1), 14: (1, 2, 0, 1), 15: (1, 1, 0, 2), 16: (4, 0, 1, 0), 17: (3, 1, 1, 0), 18: (2, 2, 1, 0), 19: (1, 3, 1, 0), 20: (0, 4, 1, 0), 21: (0, 3, 1, 1), 22: (0, 2, 1, 2), 23: (0, 1, 1, 3), 24: (0, 0, 1, 4), 25: (1, 0, 1, 3), 26: (2, 0, 1, 2), 27: (3, 0, 1, 1), 28: (2, 1, 1, 1), 29: (1, 2, 1, 1), 30: (1, 1, 1, 2), 31: (4, 0, 2, 0), 32: (3, 1, 2, 0), 33: (2, 2, 2, 0), 34: (1, 3, 2, 0), 35: (0, 4, 2, 0), 36: (0, 3, 2, 1), 37: (0, 2, 2, 2), 38: (0, 1, 2, 3), 39: (0, 0, 2, 4), 40: (1, 0, 2, 3), 41: (2, 0, 2, 2), 42: (3, 0, 2, 1), 43: (2, 1, 2, 1), 44: (1, 2, 2, 1), 45: (1, 1, 2, 2), 46: (0, 0, 2, 0)}
_RELABEL = {1: 1, 2: 6, 3: 10, 4: 13, 5: 15, 6: 14, 7: 12, 8: 9, 9: 5, 10: 4, 11: 3, 12: 2, 13: 7, 14: 11, 15: 8}
UNPRINTED = '-'

def code_tag(probe_no: int) -> str:
    """``26`` -> ``'PR26'``.

    Single source of truth for how a probe is written in figures, filenames and
    CSV columns. No zero padding: the bench sheet says PR1/PR26.
    """
    return f'PR{int(probe_no)}'

def reorder_indices() -> list[int]:
    """Index permutation of the colour grid into GMM label order.

    Returns:
        Index list ``perm`` such that grid slot ``perm[i]`` is component ``i``.
    """
    return [_RELABEL[k] - 1 for k in sorted(_RELABEL)]

def grades_by_slot(color_grade: int=COLOR_GRADE) -> list[tuple[int, int, int]]:
    """(Ye, B, R) grade triples in GMM label order, one per colour-grid slot.

    Args:
        color_grade: Number of colour grades (5 for PRISM46).

    Returns:
        One ``(ye, b, r)`` triple per component; index i == label offset i.
    """
    grid = [c for c in itertools.product(range(color_grade), repeat=3) if sum(c) == color_grade - 1]
    return [grid[j] for j in reorder_indices()]

def build_label_to_probe(glayer: int=GLAYER, color_grade: int=COLOR_GRADE) -> dict[int, int]:
    """Resolve every GMM label to the probe number sharing its colour grades.

    Derived, not assumed: the label's grade tuple is looked up in
    ``PROBE_GRADES``. On PRISM46 the result is the identity map for labels
    1..45, but deriving it is what makes that checkable instead of a coincidence.

    Args:
        glayer: Number of Cy3 layers (3 for PRISM46).
        color_grade: Number of colour grades.

    Returns:
        ``{label: probe_no}`` for labels 1 .. glayer * n_slots.

    Raises:
        KeyError: If a component's grade tuple has no probe, i.e. the panel and
            the barcode catalogue disagree.
        ValueError: If the resulting map is not injective.
    """
    grade_to_probe = {g: p for p, g in PROBE_GRADES.items()}
    slots = grades_by_slot(color_grade)
    q_num = len(slots)
    label_to_probe: dict[int, int] = {}
    for layer in range(glayer):
        for i, (ye, b, r) in enumerate(slots):
            label = layer * q_num + i + 1
            key = (r, ye, layer, b)
            if key not in grade_to_probe:
                raise KeyError(f'label {label} has grade tuple {key} with no PRISM_46plex_v1 barcode; panel/backbone mismatch.')
            label_to_probe[label] = grade_to_probe[key]
    if len(set(label_to_probe.values())) != len(label_to_probe):
        raise ValueError('label -> probe map is not injective; check reorder_indices')
    return label_to_probe

def display_name(probe_no: int, gene: str) -> str:
    """How a codeword is written in a figure or a filename.

    An unprinted codeword has no gene name, so it is written by its probe number
    alone -- ``PR2_unprinted`` -- which keeps it visible in the per-probe QC
    instead of collapsing several of them onto one ``-`` label.
    """
    return f'{code_tag(probe_no)}_unprinted' if gene == UNPRINTED else gene
