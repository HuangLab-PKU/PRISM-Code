# Automatic pipeline QC

Successful file-producing runs now require two report pairs in the readout directory:

- `readout_qc.png` and `readout_qc.json`.
- `gene_calling_qc.png` and `gene_calling_qc.json`.

The shared readout command, PR31/PR46 production drivers, their calling-only entries
and the single-acquisition VIS driver generate these automatically. The generic
classifier's `run_full_pipeline(..., output_dir=...)` also generates calling QC in
its output directory. There is no separate figure switch to remember. Missing,
corrupt or mismatched figures prevent successful stage/completion receipts.
Manual colour-space/layer figures remain available and are not replaced by QC.

## Readout

The plot preserves actual source channels and acquisition-cycle labels rather than
assuming a sequencing protocol. It shows the spatial count distribution, full-row
raw-intensity histograms and pooled log-intensity correlations. True detector and
exact/spatial deduplication counts are included when the producer recorded them;
external-table imports explicitly leave unavailable funnel stages unknown.
The final histogram bin includes values above65535, with explicit overflow counts;
this display bound never clips the input or changes classification.
Correlation uses at most 200,000 deterministic bottom-hash rows across the complete
table. The source-row count and sampling method are in the report. Histograms,
spatial counts and the detection funnel use all rows. Pooled channel correlation
is not an estimate of dye crosstalk.

## Gene calling

The composite follows the established SPRINTseq diagnostic structure: calling
outcomes, posterior distributions, entropy, spatial distribution, fitting trace,
feature abundance, posterior-threshold retention, posterior/entropy relationships
and the top-two margin. It does not import SPRINTseq or assume its cycle layout.

Absent confidence, entropy or fitting objectives are marked unavailable. In
particular, the PR31/PR46 pure-Cy3 path has no posterior; it is never imputed as
zero or included in the denominator of posterior-only retention. Imported tables
containing only emitted calls keep the missing readout rows labeled `Not emitted`,
not invented as Background. Declared unprinted features remain non-target controls.
Zero-call panel members remain visible in report counts.

Posterior-derived Q values are model diagnostics, not an empirical error-rate
calibration. A complete QC report does not establish biological specificity or
error correction for a nonredundant hybridization codebook.

## Recovery and existing data

Regenerate figures from completed tables without repeating detection or fitting:

```text
python -m prism.qc --run-dir /path/to/RUN_ID_processed --readout --gene-calling
```

Either report type can be selected alone. Existing nonstandard imported readout QC
is preserved under `readout/qc_previous/` when calling-only execution needs the
standard report. Accepted standard QC is integrity-checked rather than silently
replaced during resume.

For VIS, numerical outputs are checkpointed before plotting. A failed plot leaves
the mapping tables/model intact but does not write successful final validation.
After addressing the plotting error, use the exact original config:

```text
python -m prism.vis_pipeline --run-id RUN_ID --config decode.yaml --qc-only
```

This checks source/config/panel/count identities, refuses an active decode lock,
and regenerates QC without rerunning the decoder. It cannot silently reassign
genes after a changed recipe. PNG/report publication is atomic per file and the
report binds the figure hash; validators reject a mismatched pair.

Aggregation is chunked and the plotting sample is bounded. Normal pipeline calls
use their existing in-memory handoff; backfill streams CSVs. Neither performs an
extra full-table read solely to verify already-computed routine numerical output.

PR31 writes its GMM assignment to mapping_gmm.csv. VIS writes mapping_postcode.csv.
Both use index, Y, X, Gene and Probability. PR31 preserves its method-specific
fields and missing confidence values. PR31 preprocessing tables are stored in
processing/prism31/ alongside the readout directory.

When a stitched mosaic is available, install the shared spatial-cells package
in the pipeline environment. Both entries call its density, density QC and
Thermal ImageJ stack writers after classification. The default readout products
are density_Q20/, density_Q20_qc.json, density_Q20_qc.png and density_Q20_all.tif.
The threshold is strictly P > 0.99 for target genes with reported confidence.
