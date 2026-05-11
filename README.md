# REACHER — **R**aw image **E**xtraction **A**nd **C**alibration for **H**igh-contrast **E**xoplanet **R**esearch*

A Python pipeline for pre-processing high-contrast imaging (HCI) data
from **VLT/SPHERE-IRDIS**, designed to feed circumstellar-disk and
exoplanet searches with science-ready master cubes and parallactic
angles. The output is intended for stellar PSF subtraction using e.g. PCA via
[VIP HCI](https://github.com/vortex-exoplanet/VIP) for downstream
ADI / SDI post-processing. But can also used with other packages/methods. Note that the parallactic angle should be considered as de-rotation angele.

(* please not that this pipeline is under development. It has been tested, however outputs should not be blindly trusted.)

---

## Quick start

```python
# run_REACHER.py — edit the values below, then `python run_REACHER.py`
import logging
import os
import REACHER

config = REACHER.ReacherConfig(
    paths=REACHER.PathConfig(
        raw_input_dir="/data/raw/",
        reduced_output_dir="/data/reduced/",
    ),
)
config.processing.nproc = 10                # number of cores for the multiprocessing
config.processing.overwrite = False         # True clobbers existing outputs
config.calibration.star_loc = (400, 465)    # location of the off-axis flux calibration source

REACHER.setup_logging(
    log_file=os.path.join(config.paths.reduced_output_dir, "REACHER.log"),
    level=logging.INFO,
)

# Map each science target -> reference star (None, a single star, or
# a per-obs_night dict). iter_star_targets flattens this into the
# union of keys + non-None values.
ref_cubes_dict = {"HD121617": "HD120275A"}.   # Host star + reference star, if applicable

if __name__ == "__main__":
    REACHER.make_bad_pix_masks(config)
    REACHER.make_all_masterdarks(config)
    REACHER.make_all_masterflats(config)

    for target in REACHER.iter_star_targets(
        config.paths.raw_input_dir, ref_cubes_dict,
    ):
        REACHER.pre_calibration_and_extraction(config, target)

    # [HUMAN-IN-THE-LOOP — inspect & hand-edit centers.csv files]

    for target in REACHER.iter_star_targets(
        config.paths.raw_input_dir, ref_cubes_dict,
    ):
        REACHER.centering_and_bad_pix_correction(
            config, target, ref_cubes=ref_cubes_dict,
        )
```

Full annotated example: [`run_REACHER.py`](run_REACHER.py).

---

## Pipeline stages

| # | Step                  | Class                | Purpose                                                       |
|---|-----------------------|----------------------|---------------------------------------------------------------|
| 1 | Bad-pixel + clump map | `BadPixelMaskCreator`| MAD-based individual bad-pixel detection + clump mask         |
| 2 | Master dark           | `MasterDarkCreator`  | Median-combine darks per (obs_night, exptime)                 |
| 3 | Master flat           | `MasterFlatCreator`  | Dark-subtract + per-frame normalise + LEFT/RIGHT mean         |
| 4 | File calibration      | `FileCalibrator`     | (science − dark) / flat + bad-pixel interpolation             |
| 5 | LEFT / RIGHT split    | `ChannelSplitter`    | Slice IRDIS detector into LEFT and RIGHT channels             |
| 6 | Centering preparation | `CenteringPreparer`  | Per-file QL plots + `centers.csv` placeholder                 |
| 7 | Master sky (optional) | `MasterSkyCreator`   | Median sky frame + per-file temperature provenance            |
| 8 | Flux calibration      | `FluxCalibrator`     | ND/exposure-time normalised flux reference frame              |
| 9 | Cube assembly         | `CubeAssembler`      | Dither + cosmic + center + anamorphism + parallactic angles   |

Output of step 9 (`master_cube.fits` + `parallactic_angles.txt`) is
the input to ADI / SDI post-processing.

---

## Installation

```bash
git clone https://github.com/PaulSteimle/REACHER.git
cd REACHER
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

There is no editable install yet (no `pyproject.toml`); run from the
repo root with `PYTHONPATH` set:

```bash
PYTHONPATH=$(pwd) python run_REACHER.py
PYTHONPATH=$(pwd) pytest tests/
```

Dependencies: Python 3.8, NumPy, SciPy, Astropy, pandas,
[VIP HCI](https://github.com/vortex-exoplanet/VIP), photutils,
Matplotlib (optional plotting).

---

## Input data layout

REACHER expects raw VLT/SPHERE-IRDIS FITS files from the
[ESO archive](http://archive.eso.org/eso/eso_archive_main.html),
organised by the discovery layer's directory contract:

```
raw_input_dir/
    DARK/IRDIS/<obs_night>/<filter>/*.fits
    FLAT/IRDIS/<obs_night>/<filter>/*.fits
    BACKGROUND/IRDIS/<obs_night>/<filter>/*.fits
    <star>/IRDIS/<obs_night>/<filter>/*SPHER*.fits
```

The pre-pipeline helper `REACHER.move_files_from_download_dir(...)`
(see [`regroup_input_data.py`](regroup_input_data.py)) reads an ESO
download dir, parses each file's headers and copies it into the layout
above with an optional object-name renaming dictionary.

---

## Usage — the editable run script

`run_REACHER.py` has **no CLI arguments**: edit the values in place and
run. The script flows through three stages plus an operator step
between stages 2 and 3.

### Stage 1 — calibration master products

```python
REACHER.make_bad_pix_masks(config)
REACHER.make_all_masterdarks(config)
REACHER.make_all_masterflats(config)
```

Each helper walks `raw_input_dir/DARK|FLAT|BACKGROUND/IRDIS/` and
groups by `(obs_night, filter)` (or `(obs_night, exptime)` for darks),
then builds one master product per group. Outputs go under
`reduced_output_dir/{bad_pix_masks,master_dark,master_flat}/`.

`REACHER.make_all_masterbackgrounds(config)` raises
`NotImplementedError`: the legacy wrapper was incomplete and the
silent no-op was deliberately surfaced. Sky subtraction is handled
later by `MasterSkyCreator`.

### Stage 2 — per-target pre-calibration + extraction

```python
for target in REACHER.iter_star_targets(
    config.paths.raw_input_dir, ref_cubes_dict,
):
    REACHER.pre_calibration_and_extraction(config, target)
```

`iter_star_targets` accepts either a list of star names or a
`ref_cubes_dict`-style mapping; in the dict case it flattens the
union of keys (science targets) and non-`None` values (reference
stars). It yields one `StarTarget` per `(star, obs_night, filter)`
triple, skipping ND filters by default.

`pre_calibration_and_extraction` resolves the matching bad-pix mask
and master dark / flat for the target, then runs `FileCalibrator` and
`ChannelSplitter` over every raw science file in parallel
(`config.processing.nproc` workers), followed by `CenteringPreparer`
on the CENTER frames.

### Human-in-the-loop centering check

Between stages 2 and 3 the operator inspects the quick-look figures
produced by `CenteringPreparer` and hand-edits the satellite-spot
positions in each `centers.csv`. To make this easier:

```python
REACHER.grab_all_files(config, ref_cubes_dict.keys())   # pull into one folder
# ... operator edits the centers.csv files ...
REACHER.return_csv_files(config, ref_cubes_dict.keys()) # push edits back
```

This manual centering check is **intentional**.

### Stage 3 — centering, flux, master cubes

```python
for target in REACHER.iter_star_targets(
    config.paths.raw_input_dir, ref_cubes_dict,
):
    REACHER.centering_and_bad_pix_correction(
        config, target, ref_cubes=ref_cubes_dict,
    )
```

This runs `MasterSkyCreator` (optional, when sky files exist),
`FluxCalibrator` (per flux file), and `CubeAssembler` (final master
cube + parallactic angles) for the LEFT and RIGHT channels of each
target.

### Per-target overrides via `StarTarget`

Anything that varies *per target* (rather than per run) goes on the
`StarTarget` value object, not on the config:

```python
target = REACHER.StarTarget(
    star_name="HD101412",
    obs_night="2022-10-14",
    filter_name="DB_H23",
    star_center_index=1,
    fixed_user_mastersky_left="/path/to/master_sky_left.fits",
    fixed_user_mastersky_right="/path/to/master_sky_right.fits",
)
```

`StarTarget` is frozen, so the config stays clean across the loop.

---

## Resumability

Every step writes its output to disk and calls
`PipelineStep._skip_if_exists` to skip itself when the output already
exists and `config.processing.overwrite` is `False`. A failed run can
be resumed from the last successful step without redoing the
calibration products.

---

## Output structure

```
reduced_output_dir/
├── bad_pix_masks/             # bad_pix_mask_<obs_night>.fits + IRDIS_clump_mask.fits
├── master_dark/               # master_dark_<obs_night>_<exp_time>.fits
├── master_flat/               # master_flat_<obs_night>_<filter>.fits
├── recentering/               # operator inspection folder (grab_all_files)
├── <star>/IRDIS/<obs_night>/<filter>/
│   ├── stage1/                # CALIB_*.fits (per-file dark/flat/badpix)
│   ├── stage2/                # LEFT_*/RIGHT_*.fits + centers.csv
│   ├── left/raw_files/        # left_master_cube.fits + left_parallactic_angles.txt
│   └── right/raw_files/       # right_master_cube.fits + right_parallactic_angles.txt
└── logs/                      # per-run reacher_<timestamp>.log
```

---

## Configuration

Four nested dataclasses on `ReacherConfig` (see
[`REACHER/config.py`](REACHER/config.py) for full defaults):

| Sub-config        | Knobs                                                             |
|-------------------|-------------------------------------------------------------------|
| `PathConfig`      | `raw_input_dir`, `reduced_output_dir`                             |
| `ProcessingConfig`| `nproc`, `overwrite`, `make_plots`, `skip_dither_correction`, `correct_anamorphism`, `box_radius`, `n_sigma`, plus flags consumed by `cleanup_all_folders` |
| `CalibrationConfig`| `calib_validity_threshold` (days), `star_loc`, `flux_cut_out_size`, image-shift / -rotation library + interpolation |
| `InstrumentConfig`| Instrument-coupled keywords + DPR-type vocabulary + detector geometry. `InstrumentConfig.sphere_irdis()` returns the SPHERE preset, which is also the default for `InstrumentConfig()`. Swappable for other instruments by supplying alternative `HeaderKeywords`, `DPRValues`, and `DetectorConfig` instances. |

---

## Logging & runtime output

When `setup_logging` is called at the top of the script:

- A **once-per-process banner** prints REACHER version, git
  branch + short SHA + dirty flag, run-start timestamp, host, and
  Python version. Workers spawned by `ParallelStepGroup` do **not**
  re-emit this banner and do not clobber the parent's log file.
- Per-run log file is `<reduced_output_dir>/logs/reacher_<timestamp>.log`
  (or the path passed to `setup_logging`). Console output goes to
  stderr; the log file gets DEBUG-level records.
- The four batch helpers each print a `Starting Pipeline with N step(s): …`
  line, then one INFO line per artifact (e.g.
  `Created master_dark_2022-10-14_64.0.fits in 5.83s`), followed by
  `Pipeline finished N step(s) in …`. The `FileCalibrator` step also
  prints one line per file naming the master dark / flat used.
- Per-step `[i/N] Running X` / `[i/N] X finished in …` markers are at
  DEBUG (set the `Pipeline` logger to DEBUG to surface them).
- Blank lines visually separate each step's output block in both the
  console and the log file.
- Durations are formatted as `5.83s`, `2m 5.5s`, or `1h 2m 5s`
  depending on magnitude.
- Noisy stdout / stderr from VIP-HCI routines (bad-pixel fixers etc.)
  is suppressed via a local context manager so it does not interleave
  with REACHER's logs.

### Parallelism

Steps that are independent across files —
`FileCalibrator`, `ChannelSplitter`, the cosmic-ray correction in
`CubeAssembler`, and `CubeAssembler`'s anamorphic resampling — are
parallelised across `config.processing.nproc` workers via
`ParallelStepGroup` (a `PipelineStep` that wraps N inner steps in a
`multiprocessing.Pool`). The Pool size is clamped to the number of
inner steps, and `nproc=1` falls back to in-process sequential
execution.

---

## Public API summary

```python
import REACHER

# Config
REACHER.ReacherConfig, REACHER.PathConfig, REACHER.ProcessingConfig,
REACHER.CalibrationConfig, REACHER.InstrumentConfig
REACHER.__version__

# Logging
REACHER.setup_logging
REACHER.print_banner             # auto-called by setup_logging

# Pre-pipeline file organising
REACHER.move_files_from_download_dir

# Discovery
REACHER.StarTarget                # per-target value object (frozen)
REACHER.DataLayout                # path resolver
REACHER.list_directory            # macOS-aware ls
REACHER.list_target_stars         # raw_input_dir non-calib subdirs
REACHER.iter_star_targets         # yield StarTarget per (star, night, filter)
REACHER.stars_from_ref_cubes      # flatten a ref_cubes_dict to a star list
REACHER.make_dark_DF, make_flat_DF, make_background_DF, make_star_DF
REACHER.resolve_star_calibration  # per-row calibration matcher

# Batch helpers (matching legacy script API)
REACHER.make_bad_pix_masks
REACHER.make_all_masterdarks
REACHER.make_all_masterflats
REACHER.make_all_masterbackgrounds   # raises NotImplementedError
REACHER.pre_calibration_and_extraction
REACHER.centering_and_bad_pix_correction
REACHER.grab_all_files
REACHER.return_csv_files
REACHER.cleanup_all_folders

# Pipeline + steps (custom flows)
REACHER.Pipeline, REACHER.ParallelStepGroup, REACHER.PipelineStep
REACHER.BadPixelMaskCreator, MasterDarkCreator, MasterFlatCreator,
REACHER.FileCalibrator, ChannelSplitter, CenteringPreparer,
REACHER.MasterSkyCreator, FluxCalibrator, CubeAssembler
```

---

## Custom pipelines (advanced)

The batch helpers above build `Pipeline` instances under the hood. To
wire a custom flow, instantiate steps directly and assemble them:

```python
pipeline = REACHER.Pipeline(config, configure_logging=False)
pipeline.add_step(REACHER.MasterDarkCreator(
    config=config,
    dark_files=[...],
    master_dark_path=".../master_dark_xyz.fits",
))
pipeline.add_step(REACHER.ParallelStepGroup(
    config=config,
    steps=[REACHER.FileCalibrator(...) for ...],
    label="FileCalibrator",
))
pipeline.run()
```

Future post-processing steps (ADI / SDI) will plug into the same
`Pipeline` class via additional `PipelineStep` subclasses — see the
`# TODO: POST-PROCESSING STEPS` block at the top of
[`REACHER/pipeline.py`](REACHER/pipeline.py).

---

## Roadmap

- **ADI / SDI post-processing steps.** New `PipelineStep` subclasses
  consuming `master_cube.fits` + `parallactic_angles.txt`.
- **Editable install.** `pyproject.toml` so `pip install -e .` works
  and `PYTHONPATH=$(pwd)` is no longer needed.
- **Optional VIP `extras_require`.** Decouple the heavy VIP-HCI
  dependency chain from the core install.
- **Multi-instrument profiles.** The instrument-config seam exists
  (`InstrumentConfig.sphere_irdis()`); add a NACO / SCExAO / IRDAP
  preset when a non-SPHERE dataset comes along.
- **Dataset-discovery refinements.** The DPR-type matching is
  set-based (`science_types`, `sky_types`, …) and parameterised on the
  instrument profile; substring-matching for `BACKGROUND`-style
  composite types could be cleaned up.

---

## Known limitations

- `MasterBackground creation` (instrumental background) is not working yet, 
  however not neccessary for usual HCI data processing. Sky subtraction
  is handled by `MasterSkyCreator`.
- Anamorphic correction always uses `opencv` + `lanczos4` because
  `vip-fft` does not support non-equal axis scaling.
- The header-value scrubbing inside `move_files_from_download_dir`
  (the `DARK,LAMP`/`DARK,` etc. token cleanup) is hard-coded for
  SPHERE-style DPR strings.
- VIP version pinning: `cube_fix_badpix_interp` signature varies
  between versions, and `verbose=` is unsupported in vip-hci ≥1.6.
- No editable install yet (no `pyproject.toml`); run with
  `PYTHONPATH=$(pwd)` from the repo root.

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Issues / questions: open a GitHub issue or contact
[pasteimle@mpia.de](mailto:pasteimle@mpia.de).
