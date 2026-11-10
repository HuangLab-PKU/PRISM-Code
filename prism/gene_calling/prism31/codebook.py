"""PR31 physical slot grades; pure Cy3 is31, never PRISM46 slot31."""
from ..prism46.codebook import (COLOR_GRADE, code_tag, grades_by_slot, display_name, UNPRINTED)
from ..prism46.codebook import PROBE_GRADES as _PR46, build_label_to_probe as _build
GLAYER = 2
PURE_G_PROBE = 31
PROBE_GRADES = {p: g for p, g in _PR46.items() if p <= 30}
PROBE_GRADES[31] = (0, 0, 1, 0)
def build_label_to_probe():
    return _build(glayer=2)
