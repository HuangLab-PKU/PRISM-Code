"""
Generate PRISM codebook for PoSTcode-based gene calling.

This module generates the theoretical codebook for PRISM30 panel:
- 15 color combinations (ch1:ch2:ch4 ratios, sum=1)
- 2 layer states (ch3 = 0 or 1)
- 30 genes (15 × 2)
- 1 Pure_Cy3 (ch3=1, ch1=ch2=ch4=0)
- Total: 31 combinations
"""

import csv
from pathlib import Path
from typing import List, Tuple, Optional


def ring_spiral_order_unified(color_grade: int) -> List[Tuple[int, int, int]]:
    """
    Generate ring-spiral triplet indices in unified order (ch1, ch2, ch4).
    
    This is the same logic as in gmm_method.py but extracted for codebook generation.
    
    Args:
        color_grade: Number of grading levels (5 for PRISM30)
    
    Returns:
        List of (i1, i2, i4) tuples representing color combinations
    """
    S = color_grade - 1
    order = []
    for r in range(0, S + 1):
        S_prime = S - 3 * r
        if S_prime < 0:
            break
        ring_points = []
        if S_prime == 0:
            ring_points.append((0, 0, 0))
        else:
            edge_AB = [(S_prime - t, t, 0) for t in range(0, S_prime + 1)]  # ch1->ch2
            edge_BC = [(0, S_prime - t, t) for t in range(0, S_prime + 1)]  # ch2->ch4
            edge_CA = [(t, 0, S_prime - t) for t in range(0, S_prime + 1)]  # ch4->ch1
            seq = edge_AB + edge_BC + edge_CA
            seen = set()
            for p in seq:
                if p not in seen:
                    seen.add(p)
                    ring_points.append(p)
        ring_points_orig = [(a + r, b + r, c + r) for (a, b, c) in ring_points]
        order.extend(ring_points_orig)
    
    # Filter to valid combinations: sum == (color_grade - 1) and all values in range
    order = [
        t for t in order 
        if sum(t) == (color_grade - 1) and all(0 <= v <= (color_grade - 1) for v in t)
    ]
    
    # Remove duplicates
    out, seen = [], set()
    for t in order:
        if t not in seen:
            seen.add(t)
            out.append(t)
    
    return out


def indices_to_ratios(indices: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert color indices to normalized ratios (ch1/A, ch2/A, ch4/A).
    
    Args:
        indices: (i1, i2, i4) tuple representing color channel indices
    
    Returns:
        (ch1/A, ch2/A, ch4/A) normalized ratios that sum to 1
    """
    i1, i2, i4 = indices
    total = i1 + i2 + i4
    if total == 0:
        # Edge case: all zeros, return equal distribution
        return (1/3, 1/3, 1/3)
    return (i1 / total, i2 / total, i4 / total)


def generate_prism30_codebook(
    output_path: Optional[Path] = None,
    gene_name_prefix: str = "PRISM"
) -> List[dict]:
    """
    Generate PRISM30 codebook with 31 combinations.
    
    Args:
        output_path: Optional path to save codebook as CSV
        gene_name_prefix: Prefix for gene names (default: "PRISM")
    
    Returns:
        DataFrame with columns: ['Gene', 'ch1/A', 'ch2/A', 'ch3/A', 'ch4/A', 'log10_sum']
        - Gene: Gene identifier (PRISM_1 to PRISM_30, Pure_Cy3)
        - ch1/A, ch2/A, ch3/A, ch4/A: Normalized ratios
        - log10_sum: Initial value for intensity dimension (default: 1.0, will be learned)
    """
    color_grade = 5
    layer_states = [0, 1]  # ch3 states
    
    # Generate 15 color combinations
    color_combinations = ring_spiral_order_unified(color_grade)
    
    if len(color_combinations) != 15:
        raise ValueError(
            f"Expected 15 color combinations, got {len(color_combinations)}. "
            f"Combinations: {color_combinations}"
        )
    
    # Build codebook entries
    codebook_rows = []
    gene_idx = 1
    
    # 30 genes: 15 color combinations × 2 layer states
    for i1, i2, i4 in color_combinations:
        ch1_ratio, ch2_ratio, ch4_ratio = indices_to_ratios((i1, i2, i4))
        
        for ch3_state in layer_states:
            gene_name = f"{gene_name_prefix}_{gene_idx}"
            codebook_rows.append({
                'Gene': gene_name,
                'ch1/A': ch1_ratio,
                'ch2/A': ch2_ratio,
                'ch3/A': float(ch3_state),  # Fixed: 0 or 1
                'ch4/A': ch4_ratio,
                'log10_sum': 1.0  # Initial value, will be learned by PoSTcode
            })
            gene_idx += 1
    
    # Add Pure_Cy3 (special class)
    codebook_rows.append({
        'Gene': 'Pure_Cy3',
        'ch1/A': 0.0,
        'ch2/A': 0.0,
        'ch3/A': 1.0,  # Only ch3 is bright
        'ch4/A': 0.0,
        'log10_sum': 1.0
    })
    
    # Validate: should have 31 rows
    if len(codebook_rows) != 31:
        raise ValueError(f"Expected 31 codebook entries, got {len(codebook_rows)}")
    
    # Validate: ch1/A + ch2/A + ch4/A should be approximately 1.0 (within tolerance)
    for idx, row in enumerate(codebook_rows):
        if row['Gene'] != 'Pure_Cy3':
            ratio_sum = row['ch1/A'] + row['ch2/A'] + row['ch4/A']
            if abs(ratio_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Row {idx} ({row['Gene']}): ratios don't sum to 1.0: "
                    f"ch1/A={row['ch1/A']:.6f}, ch2/A={row['ch2/A']:.6f}, "
                    f"ch4/A={row['ch4/A']:.6f}, sum={ratio_sum:.6f}"
                )
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Gene', 'ch1/A', 'ch2/A', 'ch3/A', 'ch4/A', 'log10_sum']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(codebook_rows)
        
        print(f"Saved PRISM30 codebook to: {output_path}")
        print(f"Codebook entries: {len(codebook_rows)}")
        print(f"Genes: {[row['Gene'] for row in codebook_rows]}")
    
    return codebook_rows


if __name__ == "__main__":
    # Generate and save codebook
    output_dir = Path(__file__).parent.parent.parent / "configs" / "codebooks"
    output_path = output_dir / "prism30_codebook.csv"
    
    codebook = generate_prism30_codebook(output_path=output_path)
    
    print("\nPRISM30 Codebook Summary:")
    print("=" * 80)
    print(f"Total entries: {len(codebook)}")
    print(f"Color combinations: 15")
    print(f"Layer states: 2 (ch3 = 0 or 1)")
    print(f"Genes: 30 (PRISM_1 to PRISM_30)")
    print(f"Special class: 1 (Pure_Cy3)")
    print("\nFirst 5 entries:")
    for i, row in enumerate(codebook[:5]):
        print(f"  {i+1}. {row}")
    print("\nLast 5 entries:")
    for i, row in enumerate(codebook[-5:], start=len(codebook)-4):
        print(f"  {i}. {row}")
