# Plots: How to Replot and Edit

Two independent plotting systems live in this repo.  This document explains how to regenerate or tweak existing plots without re-running expensive simulation or QPU jobs.

---

## 1. ATLAS Calorimeter Plots

**Relevant files**
- `utils/atlas_plots.py` — histogram/grid plot functions
- `utils/HLF/atlasgeo.py` — geometry, feature extraction, and `evaluate_and_plot`

### Data flow

```
showers (tensors)
    └─► evaluate_and_plot(data_dict, binning_path, output_dir)   [atlasgeo.py]
            │  runs DifferentiableFeatureExtractor on all datasets
            │  wraps results in FeatureAdapter objects
            └─► make_validation_plots(...)           [atlas_plots.py]  ← dynamic bins
            OR  make_validation_plots_fixed(...)     [atlas_plots.py]  ← explicit ranges
                    └─► plot_atlas_style_multi(...)
                            │  saves   output_dir/<stem>.png
                            └─► output_dir/<stem>.npz   ◄── replot source
```

Every call to `plot_atlas_style_multi` writes a companion `.npz` file next to the PNG.  That file contains:

| key | content |
|-----|---------|
| `bins` | bin edges used for the histogram |
| `data_ref` | raw reference (Ground Truth (Geant4)) samples |
| `<label>` | one array per model, keyed by its label string |

Paper-facing labels are `Ground Truth (Geant4)`, `AE reconstruction`,
`Classical RBM`, and `QPU RBM`. Replotting also translates the legacy
`Data`/`Recon`/`GPU`/`QPU` keys to these names.

### Replotting from `.npz` without re-running simulation

```python
from utils.atlas_plots import replot_from_npz

replot_from_npz(
    save_dir   = "plots/my_run",   # directory that already contains *.npz files
    output_dir = "plots/my_run/replot",  # default: save_dir/replot/
    yscale     = "log",            # override y-axis scale (None → auto-infer)
    xscale     = "linear",         # override x-axis scale
    colors     = ["red", "green"], # list of colour strings (one per model)
    linestyles = ["-", "--"],      # list of linestyle strings
    ratio_min_reference_count = 5,  # hide sparse-tail ratios below this count
    make_pdf   = True,             # compile all plots into all_plots_replot.pdf
)
```

`replot_from_npz` does two passes:
1. **Individual plots** — one figure per `.npz` file, same layout as the originals, plus a `stats.json` with KS/WD/χ².
2. **Grid plots** — automatically reconstructs the per-property layer grids (Energy, MeanEta, WidthEta, MeanPhi, WidthPhi) from the accumulated per-layer files.

Axis labels are inferred from the filename stem (e.g. `Layer2_MeanEta.npz` →
`⟨u_eta⟩_{EMB2} [mm]`), and layer labels use `PreSamplerB`, `EMB1`, `EMB2`,
`EMB3`, and `TileBar0` inside the panel below the ATLAS status stamp.

### Running with fixed bin ranges

Use `make_validation_plots_fixed` (or set `fixed_bin_ranges` in `evaluate_and_plot`) when you want reproducible axis limits across runs:

```python
from utils.atlas_plots import make_validation_plots_fixed

bin_ranges = {
    "Etot_over_Einc": (0.0, 1.2),
    "Etot":           (0.0, 5e5),
    "Energy":         (0.0, 2e5),   # applied to every layer
    "MeanEta":        (-50.0, 50.0),
    "WidthEta":       (0.0, 30.0),
    "MeanPhi":        (-50.0, 50.0),
    "WidthPhi":       (0.0, 30.0),
}

make_validation_plots_fixed(
    hlf_ref,           # FeatureAdapter for GEANT4 reference
    list_hlf_models,   # list of FeatureAdapter objects, one per model
    model_labels,      # list of label strings matching list_hlf_models
    bin_ranges=bin_ranges,
    num_bins=100,
    output_dir="plots/fixed/",
)
```

Or pass `fixed_bin_ranges` directly to `evaluate_and_plot`:

```python
from utils.HLF.atlasgeo import evaluate_and_plot

evaluate_and_plot(
    data_dict = {
        "Ground Truth (Geant4)": (showers_gt, e_inc_gt),
        "AE reconstruction": (showers_gen, e_inc_gen),
    },
    binning_path     = cfg.data.binning_path,
    output_dir       = "plots/run_01/",
    fixed_bin_ranges = bin_ranges,  # omit for dynamic binning
    num_bins         = 100,
)
```

### Tweaking individual plot style

All individual plots go through `plot_atlas_style_multi` in `atlas_plots.py:271`.  Edit that function to change:
- figure size (`figsize`), ratio panel height (`height_ratios`)
- colour/linestyle of the reference band (`fill_between`) or model lines
- stat box text format (`AtlasEvaluator.get_text`, line 230)
- ratio panel y-limits (`ax1.set_ylim`, line 349)

After editing, call `replot_from_npz` on existing `.npz` files to apply the new
style without re-running extraction. The logarithmic upper distribution keeps
all bins; the lower ratio is drawn only where the reference has at least
`ratio_min_reference_count` events (5 by default).

The four standalone panels used in Sim2Science Figure 1 are exported from the
recovered artifacts with:

```bash
python3 scripts/plot_sim2science_figure2.py \
  --ratio-min-count 5 \
  --atlas-label Internal \
  --output-dir paper/Sim2Science_Workshop/figures/internal
```

This writes `fig1a_total_response`, `fig1b_layer_fractions`,
`fig1c_emb1_centroid`, and `fig1d_emb2_width` as both SVG and PDF.  The SVGs
contain only plot-intrinsic content; panel letters, subcaptions, and the shared
model legend are composed by `paper/Sim2Science_Workshop/shower_panels.sty`.
Panels (a), (c), and (d) retain ratios; panel (b) remains one scientific panel
containing the five calorimeter-sampling axes.

The dedicated-energy EMB1/EMB2 local-moment figures are built from the
fixed-energy artifacts at 5, 50, and 250 GeV. The default is the mean figure
used in the main paper:

```bash
CONTAINER=/fast_scratch_1/triumfmlutils/containers/container_qml_v3.5.5.0.sif
apptainer exec --nv -B /path/to/CaloQuVAE:/path/to/CaloQuVAE \
  --pwd /path/to/CaloQuVAE "$CONTAINER" \
  python scripts/plot_sim2science_dedicated_width_eta.py \
    --atlas-label Internal \
    --output-dir paper/Sim2Science_Workshop/figures/internal
```

This writes the six `fig3[a-f]_*.svg` panels and matching PDFs. Use
`--observable width_eta` to write the six Appendix Figure A.5 panels. Pass
`--replot-output-root` only when the additional standard NPZ audit replots are
wanted.

To regenerate both status variants, including the layer-wise panels in Appendix
Figures A.6 and the standalone five-layer grids in A.7 and A.8, run:

```bash
python3 scripts/build_sim2science_variants.py --variant all
```

The paper's shower WDs can be regenerated from the same saved NPZ artifacts
without rerunning the shower pipeline:

```bash
python scripts/calculate_shower_wd.py
```

This writes `paper/Sim2Science_Workshop/evidence/shower_wd.json` with the
unbinned AE--Geant4, Classical--Geant4, QPU--Geant4, and QPU--Classical
distances for every plotted shower panel, together with sample counts and
native observable units. The generic `replot_from_npz` statistics also store
the direct QPU--Classical value under `_pairwise`.

The uniform-range learned-latent audit is regenerated from the frozen binary
bundle, not from the notebook or a new QPU job:

```bash
python3 scripts/plot_sim2science_latent_correlations.py \
  --atlas-label Internal \
  --output-dir paper/Sim2Science_Workshop/figures/internal
```

The renderer reads the 49,994-event posterior/GPU tensors and concatenates the
10k April 22 plus 50k April 23 QPU visible-latent tensors to reproduce the
60k set used by Figure 1. It computes correlations on the 141 learned bits,
sets the diagonal to zero, and writes PNG/PDF/SVG plus a metadata JSON file.

---

## 2. D-Wave QPU Plots

**Relevant files**
- `utils/dwave/results_io.py` — `save_result` / `load_result`
- `utils/dwave/plots.py` — one `plot_*` function per experiment type

### Data flow

```
run_*_experiment(...)           # returns a plain Python dict
    └─► save_result(result, "my_experiment", output_dir="results/dwave/")
            └─► results/dwave/my_experiment_YYYYMMDD_HHMMSS.pt

# Later, in a notebook / script:
result = load_result("results/dwave/my_experiment_20250430_120000.pt")
plot_*(result, save_path="plots/my_experiment.pdf")
```

### Saving and loading

```python
from utils.dwave.results_io import save_result, load_result, list_results

# Save after an experiment run
path = save_result(result, name="srt_comparison", output_dir="results/dwave/")

# List all saved results
for p in list_results("results/dwave/"):
    print(p)

# Load and replot
result = load_result(path)
```

Files are serialised with `torch.save` (pickle-based), so they transparently handle numpy arrays, torch tensors, and plain Python scalars.

### Replotting from a saved `.pt` file

Every `plot_*` function accepts:
- the result dict as its first positional argument
- `save_path: str | None` — writes the figure to this path when provided (directory is created automatically)
- `atlas_label: str` — text passed to `hep.atlas.label` (default `"Preliminary"`)

```python
from utils.dwave.results_io import load_result
from utils.dwave.plots import (
    plot_srt_comparison,
    plot_annealing_time_sweep,
    plot_anneal_offset_experiment,
    plot_orbit_sweep_analysis,
    plot_ga_sweep_analysis,
    plot_calibration_history,
    # ... etc.
)

result = load_result("results/dwave/srt_comparison_20250430.pt")
plot_srt_comparison(result, save_path="plots/srt_comparison_v2.pdf", atlas_label="Simulation")
```

The Sim2Science Appendix Figure A.2 diagnostics use the lightweight
`scripts/plot_sim2science_qpu_aggregation.py` module.  The SRT and orbit
renderers expose separate `*_scatter` and `*_correlations` functions so the
paper can compose the two assets with LaTeX subfigures; the legacy combined
`*_tradeoff` functions remain available for compatibility.

### Quick reference: `run_*` → `plot_*` pairing

| Experiment function | Plot function | Key result keys |
|---------------------|--------------|-----------------|
| `run_srt_aggregation_comparison` | `plot_srt_aggregation_comparison` | `classical_matrix`, `physical_srt`, `logical_srt` |
| `run_annealing_time_sweep` | `plot_annealing_time_sweep` | `annealing_times`, `per_time_results`, `classical_matrix` |
| `run_anneal_offset_experiment` | `plot_anneal_offset_experiment` | `no_offset`, `with_offset`, `classical_matrix` |
| `run_orbit_sweep` | `plot_orbit_sweep_analysis` | `sweep_results` |
| `run_ga_sweep` | `plot_ga_sweep_analysis` | `ga_results` |
| `validate_beta_heterogeneous` | `plot_energy_comparison` | `rbm_energies`, `qpu_energies`, `beta` |

### Tweaking individual plot style

Each `plot_*` function in `plots.py` is self-contained.  Common edit points:
- `figsize`, `GridSpec` ratios near the top of each function
- colour palettes — look for local `*_COLOR` constants or `bar_colors` lists
- `atlas_label` argument controls the ATLAS stamp text
- `_add_atlas_label(fig, text, y=0.965)` and `_save_fig(fig, save_path)` are module-level helpers reused by all functions; edit them to change the stamp position or DPI globally
