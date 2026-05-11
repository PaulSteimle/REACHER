REACHER status snapshot (2026-05-09)

Environment
- Python interpreter: /Users/pasteimle/miniconda3/envs/REACHER_env/bin/python (3.8.20)
- Tests are run via PYTHONPATH=/Users/pasteimle/code/REACHER plus
  /Users/pasteimle/miniconda3/envs/REACHER_env/bin/pytest from the repo root.
  (No editable install yet; setup.py is minimal and install_requires is empty.)

PHASE 2 -- COMPLETE
===================

Steps 7-15 (BadPixelMaskCreator, MasterDarkCreator, MasterFlatCreator,
FileCalibrator, ChannelSplitter, CenteringPreparer, MasterSkyCreator,
FluxCalibrator, CubeAssembler) are implemented, exported from
REACHER/steps/__init__.py, and covered by tests.

Final test results (Phase 2): 59 PASSED.


PHASE 3 -- COMPLETE
===================

Final test results (Phase 3 / full suite at end of Phase 3): 69 PASSED.

Phase 3 deliverables:
  - REACHER/utils/bad_pixel.py (interpolate_bad_pixels promotion)
  - REACHER/utils/centering.py (get_center utility, NOT a step)
  - REACHER/pipeline.py (Pipeline orchestration class)
  - tests/test_pipeline.py (10 tests)
  - run_REACHER.py slimmed to argparse + Pipeline driver
  - Legacy modules deleted: routines, wrappers, master_routines,
    params, misc, filehandler, download_data
  - README updated for the new architecture
  - Phase 3 known gap: dataset-discovery layer was missing
    (Parameters / FILEHANDLER / make_*_DF / find_calib_files chain
    retired with no immediate replacement).


PHASE 4 -- COMPLETE
===================

Final test results (full suite): 129 PASSED across 14 test files.

Per-file breakdown:
  tests/test_bad_pixel.py            4 PASS
  tests/test_dark.py                 5 PASS
  tests/test_flat.py                 5 PASS
  tests/test_calibrate.py            6 PASS
  tests/test_splitting.py            6 PASS
  tests/test_centering_prep.py       7 PASS
  tests/test_sky.py                  6 PASS
  tests/test_flux.py                10 PASS
  tests/test_cube_assembly.py       10 PASS
  tests/test_pipeline.py            10 PASS
  tests/test_discovery.py           22 PASS  (NEW)
  tests/test_batch_calibration.py    9 PASS  (NEW)
  tests/test_batch_extraction.py     5 PASS  (NEW)
  tests/test_batch_cubes.py         12 PASS  (NEW)
  tests/test_batch_inspection.py    12 PASS  (NEW)


Phase 4 deliverables:

  1. REACHER/discovery/ (new subpackage)
     - target.py: StarTarget frozen dataclass (per-target value object)
     - layout.py: DataLayout (path resolver mirroring legacy
       Parameters.* properties), list_directory helper
     - dataframes.py: make_dark_DF, make_flat_DF, make_background_DF,
       make_star_DF, list_target_stars, iter_star_targets
     - calib_resolver.py: resolve_star_calibration (single replacement
       for the 3-way duplicated find_calib_files chain), uses a regex
       date extractor so all calibration product naming conventions
       work through one helper

  2. REACHER/batch/ (new subpackage)
     - calibration.py: make_bad_pix_masks, make_all_masterdarks,
       make_all_masterflats, make_all_masterbackgrounds (raises
       NotImplementedError per Architecture Rule 7)
     - extraction.py: pre_calibration_and_extraction(config, target)
     - cubes.py: centering_and_bad_pix_correction(config, target,
       ref_cubes=None) -- handles sky / flux / cubes + ref star
       reuse-of-center semantics
     - inspection.py: grab_all_files, return_csv_files,
       cleanup_all_folders, list_directory (re-exported)

  3. REACHER/__init__.py re-exports the full Phase 4 user-facing
     surface (StarTarget, DataLayout, all batch helpers) so the script
     can do `import REACHER; REACHER.make_all_masterdarks(config)`.

  4. run_REACHER.py rewritten as an editable script:
     - No argparse / no CLI args.
     - Build a config at the top via REACHER.ReacherConfig(...).
     - Call REACHER.setup_logging(...).
     - Define ref_cubes_dict / all_stars inline.
     - if __name__ == "__main__": loops over iter_star_targets and
       calls the four batch helpers in order.
     - The legacy commented-out menu of options
       (make_all_masterbackgrounds, grab_all_files, return_csv_files,
       cleanup_all_folders) is preserved as commented code.

  5. tests/conftest.py extended with the synthetic_raw_layout fixture
     (full IRDIS-shaped DARK/FLAT/SCIENCE/FLUX/CENTER/SKY trees) used
     by all four new test files.

  6. README.md updated for the editable-script workflow + the full
     Phase 4 public API.


Module layout post-Phase 4:

  REACHER/
    __init__.py                          (Phase 4 user-facing exports)
    config.py
    exceptions.py
    instrument_constants.py
    pipeline.py
    plots.py
    irdis_filters/
    io/
      __init__.py
      calib_matcher.py
      fits_handler.py
    steps/
      __init__.py
      base.py
      bad_pixel.py
      dark.py
      flat.py
      calibrate.py
      splitting.py
      centering_prep.py
      sky.py
      flux.py
      cube_assembly.py
    utils/
      __init__.py
      bad_pixel.py
      centering.py
      cosmics.py
      filters.py
      logging.py
    discovery/                           <-- NEW (Phase 4)
      __init__.py
      target.py
      layout.py
      dataframes.py
      calib_resolver.py
    batch/                               <-- NEW (Phase 4)
      __init__.py
      calibration.py
      extraction.py
      cubes.py
      inspection.py


Public API after Phase 4 (REACHER/__init__.py):

  Pipeline, PipelineStep, setup_logging,
  ReacherConfig, PathConfig, InstrumentConfig, ProcessingConfig,
  CalibrationConfig,
  9 step classes,
  custom exceptions (ReacherError + 8 subclasses),

  Phase 4 discovery:
    StarTarget, DataLayout, list_directory, list_target_stars,
    iter_star_targets, make_dark_DF, make_flat_DF,
    make_background_DF, make_star_DF, resolve_star_calibration,

  Phase 4 batch helpers:
    make_bad_pix_masks, make_all_masterdarks, make_all_masterflats,
    make_all_masterbackgrounds (NotImplementedError),
    pre_calibration_and_extraction,
    centering_and_bad_pix_correction,
    grab_all_files, return_csv_files, cleanup_all_folders.


Design decisions worth recording (Phase 4)
==========================================

1. Per-target state lives on the StarTarget value object, NOT on
   ReacherConfig. The legacy script mutated `pars.star_name`,
   `pars.obs_night`, `pars.filter_name`,
   `pars.fixed_user_mastersky[...]` and `pars.sky_files_exist`
   inside the loop body. The Phase 4 design forbids that pattern
   (Architecture Rule 3: "no mutable state set during pipeline
   execution"). StarTarget is frozen; per-iteration overrides are
   constructed via `StarTarget(...)` or `target.with_overrides(...)`.

2. resolve_star_calibration uses a single regex-based date extractor
   instead of the legacy ad-hoc string parsing. The legacy
   find_calib_files used:
     bad_pix_mask_<date>.fits     -> f.split('_')[-1].split('.')[0]
     master_dark_<date>_<exp>.fits -> f.split('_')[-2]
     master_flat_<date>_<filt>.fits -> f.split('master_flat_')[1][:10]
   Each parser was specific to one filename pattern. Phase 4 uses
   r"\d{4}-\d{2}-\d{2}" against the basename for every product type.

3. The cube-assembly batch helper computes the recentering shift
   eagerly (calling utils.centering.get_center synchronously) before
   adding the CubeAssembler step. This mirrors the legacy
   `get_center_wrapper -> make_the_cube_wrapper` chain. The
   alternative (a centering Step that produces a shift artifact on
   disk) was considered and rejected during Phase 3 because get_center
   has no FITS output.

4. The ref-star branch reuses the science target's CENTER file +
   recentering shift; ref CubeAssembler instances pull science frames
   from the ref star's own stage2 directory. This matches the legacy
   semantics exactly.

5. cleanup_all_folders uses shutil.rmtree (Architecture Rule 5).
   The legacy code used os.system('rm -r ...'); a dedicated test
   verifies os.system is never called.

6. The conftest.py raw-layout fixture writes IRDIS-shaped (1024x2048)
   FITS files. This is necessary because BadPixelMaskCreator filters
   input darks by row-count == 1024; smaller frames would silently
   produce empty mask outputs. Tests run in ~30s on the env Python.


Phase 4 backlog (out of scope for this rebuild):
  - Post-processing (ADI / SDI) PipelineStep classes (the
    `# TODO: POST-PROCESSING STEPS` block in REACHER/pipeline.py
    is the documented insertion point).
  - Editable install (pyproject.toml / setup.py).
  - Optional: a setup.py 'extras_require' entry for vip-hci so
    `pip install reacher[hci]` pulls the heavy dependency chain.


Rebuild summary
===============

The full REACHER rebuild (Phases 1-4) is complete:

  - Phase 1: foundations (exceptions, instrument constants, config
             dataclasses, fits_handler, calib_matcher, logging).
  - Phase 2: 9 step classes (one per pipeline stage) with 59 tests.
  - Phase 3: Pipeline orchestration + tests + slim run_REACHER.py +
             legacy retirement + README.
  - Phase 4: dataset-discovery layer + batch helpers + editable
             user-facing run script + 60 new tests.

Final test count: 129 PASSED across 14 test files.


PHASE 5 -- Instrument profile abstraction (2026-05-11)
======================================================

Final test results: 154 PASSED, 8 FAILED (162 total, 14 NEW).

The 8 failures are pre-existing -- they all hit the FileCalibrator
`_apply_bad_pixel_correction` 2D-branch / `verbose=False` kwarg
introduced in a separate in-progress edit to
`REACHER/steps/calibrate.py`. None of the failures involve the
instrument-profile abstraction.

Phase 5 deliverables
--------------------

1. Typed instrument profile on `REACHER/config.py`.
   `InstrumentConfig` now bundles three frozen sub-dataclasses:

   - `HeaderKeywords`  -- 9 instrument-specific FITS header keywords
                          (filter_name, nd_filter_name, nd_filter_id,
                          dpr_type, detector_temp, parang_start,
                          parang_end, dither_x, dither_y).
   - `DPRValues`       -- 7 set-based DPR vocabularies (frozenset[str]
                          fields: science_types, sky_types, dark_types,
                          flat_types, flux_types, center_types,
                          background_substrings).
   - `DetectorConfig`  -- 5 fields (left_channel, right_channel,
                          clump_positions, clump_radius, frame_shape).

   Plus the pre-existing optical scalars (`sphere_pa_offset`,
   `sphere_true_north_correction`, `sphere_anamorphic_ratio`).

   Named factory: `InstrumentConfig.sphere_irdis()` returns the
   SPHERE/IRDIS preset. `InstrumentConfig()` with no args produces
   identical state (back-compat).

2. SPHERE constants single-source-of-truth in
   `REACHER/instrument_constants.py`. The module retains its
   module-level constants (for back-compat with existing test imports)
   and adds:

   - `IRDIS_FRAME_SHAPE = (1024, 2048)` (new constant).
   - `SPHERE_HEADER_*` keyword names.
   - `SPHERE_DPR_*` frozensets (science / sky / dark / flat / flux /
     center / background-substrings).

   The SPHERE factory pulls every value from this module, so updating
   an instrument constant only requires editing
   `instrument_constants.py`.

3. Call-site updates -- 27+ hardcoded literal references replaced
   with `config.instrument.<keys|values|detector>` reads. Breakdown:

   - REACHER/steps/sky.py            (1 module-level constant removed)
   - REACHER/steps/flux.py           (3 module-level constants removed)
   - REACHER/steps/cube_assembly.py  (5 inline string keys removed,
                                       _REQUIRED_HEADER_KEYS replaced
                                       with config-driven tuple)
   - REACHER/steps/splitting.py      (channel slices via config)
   - REACHER/steps/flat.py           (channel slices via config)
   - REACHER/steps/bad_pixel.py      (clump positions / radius /
                                       frame shape via config)
   - REACHER/discovery/dataframes.py (4 header keyword reads;
                                       new `instrument=` parameter on
                                       make_dark_DF / make_flat_DF /
                                       make_background_DF / make_star_DF;
                                       background substrings via
                                       `instrument.values.background_substrings`)
   - REACHER/batch/cubes.py          (_resolve_science_exp_time uses
                                       `science_types` set membership;
                                       make_star_DF call passes
                                       `instrument=config.instrument`)
   - REACHER/batch/extraction.py     (make_star_DF call passes
                                       `instrument=config.instrument`)
   - REACHER/batch/calibration.py    (make_dark_DF / make_flat_DF calls
                                       pass `instrument=config.instrument`)
   - REACHER/utils/filters.py        (docstrings updated to refer to
                                       config-driven keywords)

4. Tests -- `tests/test_instrument_profile.py` (NEW, 14 tests):

   - SPHERE factory matches the module constants.
   - `InstrumentConfig()` no-args equals `InstrumentConfig.sphere_irdis()`.
   - `ReacherConfig` default carries SPHERE profile.
   - HeaderKeywords / DPRValues / DetectorConfig are frozen.
   - DPR set-membership semantics.
   - MasterSkyCreator reads the configured `keys.detector_temp`.
   - MasterSkyCreator raises HeaderKeyError when the configured
     keyword is missing.
   - ChannelSplitter uses the configured `detector.left_channel` /
     `right_channel` slices.
   - BadPixelMaskCreator uses the configured `detector.clump_positions`
     / `detector.clump_radius` / `detector.frame_shape`.
   - `_resolve_science_exp_time` filters via
     `config.instrument.values.science_types`.
   - `make_dark_DF` honours the configured
     `instrument.values.background_substrings`.

Conventions
-----------

- Sub-dataclasses are `@dataclass(frozen=True)` so the config can be
  pickled across multiprocessing Pool workers safely.
- DPR-value sets are `frozenset[str]` (hashable, immutable, support
  multi-token vocabularies).
- FITS-standard keywords (`EXPTIME`, `DATE-OBS`, `DATE`, `OBJECT`) are
  deliberately NOT modelled in `HeaderKeywords` -- they are universal
  across telescopes per the FITS standard and stay hardcoded.

Substring matching (load-bearing legacy behaviour)
--------------------------------------------------

The legacy `make_dark_DF` filtered K-band BACKGROUND rows via the
substring rule:

    "BACKGROUND" in header["HIERARCH ESO DPR TYPE"]
    and "K" in header["HIERARCH ESO INS COMB IFLT"]

The substring (rather than equality) match is preserved because the
legacy data has composite DPR strings like `"DARK,BACKGROUND"` that
must match. Substrings are stored in
`InstrumentConfig.values.background_substrings` (`frozenset[str]`) so
the legacy SPHERE behaviour is the default but the rule is
swappable. The substring-match contract is documented inline in
`make_dark_DF`.

Phase 5 backlog
---------------

- Header-value scrubbing inside `REACHER/utils/download.py` (tokens
  like `DARK,BACKGROUND` / `DARK,` / `,LAMP` / `LAMP,`) was left inline.
  The download utility is a pre-pipeline organiser, not a pipeline
  step, and its rules are tied to the ESO archive structure; making
  them swappable was deemed out of scope by user request.
