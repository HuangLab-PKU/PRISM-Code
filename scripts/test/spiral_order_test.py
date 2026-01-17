import itertools
import math
import sys
from typing import List, Tuple


def generate_layered_order(color_grade: int) -> List[Tuple[int, int, int]]:
    """
    Legacy layered order: all (i,j,k) with i+j+k=S where S=color_grade-1,
    produced by itertools.product filtered by sum==S.
    i corresponds to ch2 bin index, j to ch4, k to ch1 in the legacy centroid code.
    """
    S = color_grade - 1
    combos = itertools.product(range(color_grade), repeat=3)
    layered = [t for t in combos if sum(t) == S]
    return layered


def barycentric_to_cartesian(points: List[Tuple[int, int, int]]) -> List[Tuple[float, float]]:
    """
    Map integer barycentric (a,b,c) with a+b+c=S to 2D for an equilateral triangle.
    Triangle corners: A(0,0), B(1,0), C(0.5, sqrt(3)/2). Use normalized weights.
    """
    coords = []
    sqrt3_2 = math.sqrt(3) / 2.0
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, sqrt3_2)

    # Sum S from first point
    if not points:
        return coords
    S = sum(points[0])
    for (a, b, c) in points:
        wa = a / S
        wb = b / S
        wc = c / S
        x = wa * A[0] + wb * B[0] + wc * C[0]
        y = wa * A[1] + wb * B[1] + wc * C[1]
        coords.append((x, y))
    return coords


def ring_spiral_order(color_grade: int, start: str = "ch1") -> List[Tuple[int, int, int]]:
    """
    Generate spiral from outer to inner rings for sum S=color_grade-1.
    Outer ring: min(a,b,c)=0; next ring: min=1; ... until center.
    Within each ring, traverse boundary starting from a corner corresponding to 'start'
    in order ch1 -> ch2 -> ch4 (C -> A -> B) along triangle edges.

    start: one of {"ch1","ch2","ch4"}
    Returns ordered list of (a,b,c) with a+b+c=S.
    """
    S = color_grade - 1
    order = []

    # Corner mapping in (a',b',c') space after subtracting ring r
    # We interpret original (a,b,c) as (ch2, ch4, ch1)
    # Corners in reduced space: A'(a'=S',b'=0,c'=0)->ch2; B'(0,S',0)->ch4; C'(0,0,S')->ch1
    start_corner = {
        "ch1": "C",
        "ch2": "A",
        "ch4": "B",
    }[start]

    for r in range(0, (S // 1) + 1):
        S_prime = S - 3 * r
        if S_prime < 0:
            break
        # Reduced boundary points in (a',b',c') with sum S' and at least one zero
        ring_points = []
        if S_prime == 0:
            # Center point (a',b',c')=(0,0,0) → (r,r,r)
            ring_points.append((0, 0, 0))
        else:
            # Edges in order C'→A', then A'→B', then B'→C'
            # C'→A': b'=0, a' from 0..S', c'=S'-a'
            edge_CA = [(a_, 0, S_prime - a_) for a_ in range(0, S_prime + 1)]
            # A'→B': c'=0, b' from 0..S', a'=S'-b'
            edge_AB = [(S_prime - b_, b_, 0) for b_ in range(0, S_prime + 1)]
            # B'→C': a'=0, c' from 0..S', b'=S'-c'
            edge_BC = [(0, S_prime - c_, c_) for c_ in range(0, S_prime + 1)]

            # Concatenate with corner start
            if start_corner == "C":
                seq = edge_CA + edge_AB + edge_BC
            elif start_corner == "A":
                seq = edge_AB + edge_BC + edge_CA
            else:  # "B"
                seq = edge_BC + edge_CA + edge_AB

            # Remove duplicated corner points at junctions while preserving order
            seen = set()
            ring_points = []
            for p in seq:
                if p not in seen:
                    seen.add(p)
                    ring_points.append(p)

        # Map back to (a,b,c) by adding r
        ring_points_orig = [(a + r, b + r, c + r) for (a, b, c) in ring_points]
        order.extend(ring_points_orig)

    # Filter to ensure a+b+c==S and each component in [0,color_grade-1]
    order = [t for t in order if sum(t) == S and all(0 <= v <= S for v in t)]
    # Deduplicate while preserving order
    out = []
    seen = set()
    for t in order:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def spiral_order(points: List[Tuple[int, int, int]], direction: str = "CW") -> List[int]:
    """
    Compute a spiral-like order from center outward on the 2D embedding.
    Strategy: sort by (radius, angle), where radius from centroid, angle from arctan2.
    Returns indices into the original points list (0-based).
    """
    coords = barycentric_to_cartesian(points)
    # centroid
    cx = sum(x for x, _ in coords) / len(coords)
    cy = sum(y for _, y in coords) / len(coords)

    # Define triangle vertices consistent with barycentric_to_cartesian
    sqrt3_2 = math.sqrt(3) / 2.0
    A = (0.0, 0.0)           # ch2
    B = (1.0, 0.0)           # ch4
    C = (0.5, sqrt3_2)       # ch1
    # Reference angle set to vector from centroid to C (ch1)
    theta_C = math.atan2(C[1] - cy, C[0] - cx)

    polar = []
    for idx, (x, y) in enumerate(coords):
        dx = x - cx
        dy = y - cy
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        # Normalize relative to C and choose direction
        theta_rel = (theta - theta_C) % (2 * math.pi)
        if direction.upper() == "CW":
            theta_use = (2 * math.pi - theta_rel) % (2 * math.pi)
        else:
            theta_use = theta_rel
        polar.append((r, theta_use, idx))

    # stable sort: by radius ascending, then angle ascending
    polar.sort(key=lambda t: (round(t[0], 6), t[1]))
    return [idx for _, __, idx in polar]


def main():
    # Accept optional CLI arg: color_grade
    if len(sys.argv) > 1:
        try:
            color_grade = int(sys.argv[1])
        except Exception:
            color_grade = 5
    else:
        color_grade = 5  # default
    layered = generate_layered_order(color_grade)
    # Ring-based outer→inner, starting at ch1 then ch2→ch4 along edges
    ring_spiral = ring_spiral_order(color_grade, start="ch1")
    # Also compute previous polar-based for comparison
    order_spiral_idx = spiral_order(layered, direction="CW")

    # 1-based labels
    layered_labels = list(range(1, len(layered) + 1))

    # Build mapping for ring_spiral: layered (1-based) -> ring_spiral rank (1-based)
    idx_map_ring = {tuple(t): i for i, t in enumerate(layered)}
    layered_to_ring = {i + 1: (ring_spiral.index(layered[i]) + 1) for i in range(len(layered))}

    print("color_grade=", color_grade, "(S=color_grade-1)")
    print("Total points:", len(layered))
    print("Layered order (first to last):")
    print(layered)
    print("Ring-spiral order (outer→inner, ch1→ch2→ch4) (a,b,c):")
    print(ring_spiral)
    print("Layered->RingSpiral (1-based mapping):")
    print(layered_to_ring)
    
    # Previous polar-based (for reference)
    print("Polar-spiral index order (0-based):")
    print(order_spiral_idx)
    print("Points in polar-spiral order (a,b,c):")
    print([layered[i] for i in order_spiral_idx])


if __name__ == "__main__":
    main()


