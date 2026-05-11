########################################################################################################
#######                                                                                          #######
####### This file is an example script to run the REACHER pipeline on a multiple target survey. #######
####### Edit the values in the "USER PARAMETERS" block, then run:                                #######
#######                                                                                          #######
#######     python run_REACHER.py                                                                #######
#######                                                                                          #######
####### No CLI arguments. The pipeline reads its inputs from `config` below.                     #######
########################################################################################################

import logging
import os

import REACHER

# ----------------------------------------------------------------------
# USER PARAMETERS -- edit these in place, then run the script.
# ----------------------------------------------------------------------
# Build a fresh ReacherConfig instance. The dataclasses come pre-loaded
# with sensible defaults (see REACHER/config.py); override the fields
# you care about by assignment.

config = REACHER.ReacherConfig(
    paths=REACHER.PathConfig(
        raw_input_dir="/Users/pasteimle/SPHERE_test_data/raw_files",
        reduced_output_dir="/Users/pasteimle/SPHERE_test_data/reduced_cubes",
    ),
    instrument=REACHER.InstrumentConfig(),
    processing=REACHER.ProcessingConfig(),
    calibration=REACHER.CalibrationConfig(),
)

# Processing knobs that the legacy script set on `pars.*`.
config.processing.nproc = 10
config.processing.overwrite = True
config.processing.skip_dither_correction = True
config.processing.save_fullsize_flux = False
config.processing.only_flux = False
config.processing.full_output = False
config.calibration.star_loc = (400, 465)

# Optional cleanup flags read by REACHER.cleanup_all_folders().
config.processing.clean_calibrations = False
config.processing.clean_stage_1_and_2 = False

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
REACHER.setup_logging(
    log_file=os.path.join(config.paths.reduced_output_dir, "REACHER.log"),
    level=logging.INFO,
)

# ----------------------------------------------------------------------
# REFERENCE-CUBE MAPPING + STAR LIST
# ----------------------------------------------------------------------
# Map science star -> reference star (or per-obs_night dict, or None).
# See REACHER/batch/cubes.py::_resolve_ref_stars for the full schema.
# `iter_star_targets` flattens this dict into the union of keys + values
# (skipping None), so both science targets and reference stars get
# pre-calibrated and extracted in Stage 2.
ref_cubes_dict = {"HD121617": "HD120275A"}


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Stage 1: calibration master products ----
    # Uncomment the lines you want to run. Re-running with
    # processing.overwrite=False acts like the legacy 'resume' mode
    # (skip-if-exists).

    # REACHER.make_bad_pix_masks(config)
    # REACHER.make_all_masterdarks(config)
    # REACHER.make_all_masterflats(config)

    # The master-background workflow was incomplete in the legacy code.
    # Calling this raises NotImplementedError. Sky subtraction is handled
    # by REACHER.MasterSkyCreator (Stage 3a in centering_and_bad_pix_correction).
    # _ = REACHER.make_all_masterbackgrounds(config)

    # ---- Stage 2: per-target pre-calibration + extraction ----
    # for target in REACHER.iter_star_targets(
    #     config.paths.raw_input_dir, ref_cubes_dict,
    # ):
    #     REACHER.pre_calibration_and_extraction(config, target)

    # ---- HUMAN-IN-THE-LOOP centering check ----
    # Pull the auto-generated centers.csv into a flat folder for the
    # operator to inspect / hand-edit, then push the edited files back.
    # Uncomment when you reach the centering review:
    # _ = REACHER.grab_all_files(config, ref_cubes_dict.keys())
    # _ = REACHER.return_csv_files(config, ref_cubes_dict.keys())

    # ---- Stage 3: centering + flux + master cubes ----
    
    for star in ref_cubes_dict.keys():
        # Per-target overrides go on the StarTarget value object, not on
        # the config. Build the target inside the loop body so each
        # iteration starts from a known state.

        # if star == "HD101412":
        #     user_sky_left = (
        #         "/nexus/.../master_sky_HD101412_left.fits"
        #     )
        #     user_sky_right = (
        #         "/nexus/.../master_sky_HD101412_right.fits"
        #     )
        # else:
        #     user_sky_left = None
        #     user_sky_right = None

        star_parent_dir = os.path.join(
            config.paths.raw_input_dir, star, "IRDIS",
        )
        all_obs_nights = REACHER.list_directory(star_parent_dir)

        for obs_night in all_obs_nights:
            obs_night_parent_dir = os.path.join(star_parent_dir, obs_night)
            filters = REACHER.list_directory(obs_night_parent_dir)

            for filter_name in filters:
                target = REACHER.StarTarget(
                    star_name=star,
                    obs_night=obs_night,
                    filter_name=filter_name,
                    # fixed_user_mastersky_left=user_sky_left,
                    # fixed_user_mastersky_right=user_sky_right,
                )
                REACHER.centering_and_bad_pix_correction(
                    config, target, ref_cubes=ref_cubes_dict,
                )

    # ---- Optional cleanup ----
    # REACHER.cleanup_all_folders(config)
