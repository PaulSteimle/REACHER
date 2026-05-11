"""Tests for the typed instrument-profile abstraction.

The ``InstrumentConfig`` carries three nested sub-configurations
(:class:`HeaderKeywords`, :class:`DPRValues`, :class:`DetectorConfig`)
that together define the instrument-coupling surface of REACHER.
These tests cover:

* The SPHERE factory produces the legacy defaults.
* The no-arg constructor equals the SPHERE factory (back-compat).
* The nested dataclasses are frozen (so they pickle across MP workers).
* An alternate ``HeaderKeywords`` profile routes through the steps as
  expected (the step calls ``header.get(<configured key>)``).
* An alternate ``DetectorConfig`` swaps the channel slices.
* An alternate ``DPRValues`` swaps the science-type vocabulary.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from REACHER.config import (
    DetectorConfig,
    DPRValues,
    HeaderKeywords,
    InstrumentConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.exceptions import HeaderKeyError
from REACHER.IRDIS_instrument_constants import (
    IRDIS_CLUMP_POSITIONS,
    IRDIS_CLUMP_RADIUS,
    IRDIS_FRAME_SHAPE,
    IRDIS_LEFT_CHANNEL,
    IRDIS_RIGHT_CHANNEL,
    SPHERE_ANAMORPHIC_RATIO,
    SPHERE_DPR_BACKGROUND_SUBSTRINGS,
    SPHERE_DPR_CENTER_TYPES,
    SPHERE_DPR_DARK_TYPES,
    SPHERE_DPR_FLAT_TYPES,
    SPHERE_DPR_FLUX_TYPES,
    SPHERE_DPR_SCIENCE_TYPES,
    SPHERE_DPR_SKY_TYPES,
    SPHERE_HEADER_DETECTOR_TEMP,
    SPHERE_HEADER_DITHER_X,
    SPHERE_HEADER_DITHER_Y,
    SPHERE_HEADER_DPR_TYPE,
    SPHERE_HEADER_FILTER_NAME,
    SPHERE_HEADER_ND_FILTER_ID,
    SPHERE_HEADER_ND_FILTER_NAME,
    SPHERE_HEADER_PARANG_END,
    SPHERE_HEADER_PARANG_START,
    SPHERE_PA_OFFSET,
    SPHERE_TRUE_NORTH_CORRECTION,
)


# ---------------------------------------------------------------------------
# Factory equivalence
# ---------------------------------------------------------------------------
def test_sphere_irdis_factory_matches_module_constants():
    """The factory wires every SPHERE module-level constant through."""
    inst = InstrumentConfig.sphere_irdis()

    # Optical scalars.
    assert inst.sphere_pa_offset == SPHERE_PA_OFFSET
    assert inst.sphere_true_north_correction == SPHERE_TRUE_NORTH_CORRECTION
    assert inst.sphere_anamorphic_ratio == SPHERE_ANAMORPHIC_RATIO

    # Header keywords.
    assert inst.keys.filter_name == SPHERE_HEADER_FILTER_NAME
    assert inst.keys.nd_filter_name == SPHERE_HEADER_ND_FILTER_NAME
    assert inst.keys.nd_filter_id == SPHERE_HEADER_ND_FILTER_ID
    assert inst.keys.dpr_type == SPHERE_HEADER_DPR_TYPE
    assert inst.keys.detector_temp == SPHERE_HEADER_DETECTOR_TEMP
    assert inst.keys.parang_start == SPHERE_HEADER_PARANG_START
    assert inst.keys.parang_end == SPHERE_HEADER_PARANG_END
    assert inst.keys.dither_x == SPHERE_HEADER_DITHER_X
    assert inst.keys.dither_y == SPHERE_HEADER_DITHER_Y

    # DPR vocabularies.
    assert inst.values.science_types == SPHERE_DPR_SCIENCE_TYPES
    assert inst.values.sky_types == SPHERE_DPR_SKY_TYPES
    assert inst.values.dark_types == SPHERE_DPR_DARK_TYPES
    assert inst.values.flat_types == SPHERE_DPR_FLAT_TYPES
    assert inst.values.flux_types == SPHERE_DPR_FLUX_TYPES
    assert inst.values.center_types == SPHERE_DPR_CENTER_TYPES
    assert (
        inst.values.background_substrings == SPHERE_DPR_BACKGROUND_SUBSTRINGS
    )

    # Detector geometry.
    assert inst.detector.left_channel == IRDIS_LEFT_CHANNEL
    assert inst.detector.right_channel == IRDIS_RIGHT_CHANNEL
    assert inst.detector.clump_positions == IRDIS_CLUMP_POSITIONS
    assert inst.detector.clump_radius == IRDIS_CLUMP_RADIUS
    assert inst.detector.frame_shape == IRDIS_FRAME_SHAPE


def test_default_instrument_config_equals_sphere_factory():
    """``InstrumentConfig()`` with no args equals the SPHERE factory.

    Back-compatibility guarantee: the legacy code paths that did
    ``InstrumentConfig()`` keep working.
    """
    default = InstrumentConfig()
    factory = InstrumentConfig.sphere_irdis()

    assert default.sphere_pa_offset == factory.sphere_pa_offset
    assert (
        default.sphere_true_north_correction
        == factory.sphere_true_north_correction
    )
    assert default.sphere_anamorphic_ratio == factory.sphere_anamorphic_ratio
    assert default.keys == factory.keys
    assert default.values == factory.values
    assert default.detector == factory.detector


def test_reacher_config_default_carries_sphere_profile(tmp_path):
    """``ReacherConfig`` no-args yields a SPHERE-profiled config."""
    config = ReacherConfig(
        paths=PathConfig(
            raw_input_dir=str(tmp_path), reduced_output_dir=str(tmp_path),
        ),
    )
    assert config.instrument.keys.filter_name == SPHERE_HEADER_FILTER_NAME
    assert config.instrument.values.science_types == SPHERE_DPR_SCIENCE_TYPES


# ---------------------------------------------------------------------------
# Frozen-ness (immutability across MP boundaries)
# ---------------------------------------------------------------------------
def test_header_keywords_is_frozen():
    keys = HeaderKeywords(
        filter_name="A", nd_filter_name="B", nd_filter_id="C",
        dpr_type="D", detector_temp="E", parang_start="F",
        parang_end="G", dither_x="H", dither_y="I",
    )
    with pytest.raises(FrozenInstanceError):
        keys.filter_name = "X"  # type: ignore[misc]


def test_dpr_values_is_frozen():
    v = DPRValues(
        science_types=frozenset({"OBJECT"}),
        sky_types=frozenset(),
        dark_types=frozenset(),
        flat_types=frozenset(),
        flux_types=frozenset(),
        center_types=frozenset(),
        background_substrings=frozenset(),
    )
    with pytest.raises(FrozenInstanceError):
        v.science_types = frozenset()  # type: ignore[misc]


def test_detector_config_is_frozen():
    d = DetectorConfig(
        left_channel={"rows": slice(0, 1), "cols": slice(0, 1), "shape": (1, 1)},
        right_channel={"rows": slice(0, 1), "cols": slice(0, 1), "shape": (1, 1)},
        clump_positions=[],
        clump_radius=1,
        frame_shape=(1, 1),
    )
    with pytest.raises(FrozenInstanceError):
        d.clump_radius = 999  # type: ignore[misc]


def test_dpr_set_membership_for_science_type():
    """Confirm the DPR-value sets implement set-membership semantics."""
    inst = InstrumentConfig()
    assert "OBJECT" in inst.values.science_types
    assert "SKY" not in inst.values.science_types
    assert "OBJECT,FLUX" in inst.values.flux_types
    assert "OBJECT,CENTER" in inst.values.center_types


# ---------------------------------------------------------------------------
# Alternate-profile routing through a step
# ---------------------------------------------------------------------------
def _make_config_with_temp_key(tmp_path: Path, temp_key: str) -> ReacherConfig:
    """Build a ReacherConfig whose detector_temp keyword is overridden."""
    raw = tmp_path / "raw"
    reduced = tmp_path / "reduced"
    raw.mkdir(parents=True, exist_ok=True)
    reduced.mkdir(parents=True, exist_ok=True)

    paths = PathConfig(raw_input_dir=str(raw), reduced_output_dir=str(reduced))

    # Build a custom keys profile by replacing the temp key.
    sphere = InstrumentConfig.sphere_irdis()
    custom_keys = HeaderKeywords(
        filter_name=sphere.keys.filter_name,
        nd_filter_name=sphere.keys.nd_filter_name,
        nd_filter_id=sphere.keys.nd_filter_id,
        dpr_type=sphere.keys.dpr_type,
        detector_temp=temp_key,
        parang_start=sphere.keys.parang_start,
        parang_end=sphere.keys.parang_end,
        dither_x=sphere.keys.dither_x,
        dither_y=sphere.keys.dither_y,
    )
    custom_instrument = InstrumentConfig(
        sphere_pa_offset=sphere.sphere_pa_offset,
        sphere_true_north_correction=sphere.sphere_true_north_correction,
        sphere_anamorphic_ratio=sphere.sphere_anamorphic_ratio,
        keys=custom_keys,
        values=sphere.values,
        detector=sphere.detector,
    )
    return ReacherConfig(
        paths=paths,
        instrument=custom_instrument,
        processing=ProcessingConfig(overwrite=True, make_plots=False),
    )


def test_master_sky_creator_reads_configured_temp_key(tmp_path):
    """A non-SPHERE detector_temp keyword routes through MasterSkyCreator.

    Demonstrates that overriding ``InstrumentConfig.keys.detector_temp``
    causes the step to call ``header.get(<configured key>)`` rather
    than the SPHERE default.
    """
    from REACHER.steps.sky import MasterSkyCreator

    config = _make_config_with_temp_key(tmp_path, temp_key="MYINSTR TEMP")

    # Write a sky file whose header uses the configured keyword name.
    sky_data = np.full((5, 5), 1.0, dtype=np.float32)
    h = fits.Header()
    h["EXPTIME"] = 1.0
    h["MYINSTR TEMP"] = 300.0
    sky_path = tmp_path / "sky.fits"
    fits.writeto(sky_path, sky_data, header=h, overwrite=True)

    master_sky = tmp_path / "master_sky.fits"
    step = MasterSkyCreator(
        config=config,
        sky_files=[sky_path],
        master_sky_path=master_sky,
    )
    step.run()
    assert master_sky.exists()

    # The step should have routed the temperature off the custom keyword:
    # MEANTEMP = 300.0.
    out_hdr = fits.getheader(master_sky)
    assert out_hdr["MEANTEMP"] == 300.0


def test_master_sky_creator_missing_configured_temp_key_raises(tmp_path):
    """When the configured keyword is absent, the step raises HeaderKeyError."""
    from REACHER.steps.sky import MasterSkyCreator

    config = _make_config_with_temp_key(tmp_path, temp_key="MYINSTR TEMP")

    sky_data = np.full((5, 5), 1.0, dtype=np.float32)
    h = fits.Header()
    h["EXPTIME"] = 1.0
    # Deliberately do not set MYINSTR TEMP.
    sky_path = tmp_path / "sky.fits"
    fits.writeto(sky_path, sky_data, header=h, overwrite=True)

    master_sky = tmp_path / "master_sky.fits"
    step = MasterSkyCreator(
        config=config,
        sky_files=[sky_path],
        master_sky_path=master_sky,
    )
    with pytest.raises(HeaderKeyError):
        step.run()


# ---------------------------------------------------------------------------
# Detector-config swap
# ---------------------------------------------------------------------------
def test_channel_splitter_uses_configured_detector_slices(tmp_path):
    """A custom DetectorConfig drives ChannelSplitter's slice picks.

    We swap the LEFT/RIGHT channel slices to small custom regions and
    verify the splitter pulls from those regions in the output FITS.
    """
    from REACHER.steps.splitting import ChannelSplitter

    raw = tmp_path / "raw"
    reduced = tmp_path / "reduced"
    raw.mkdir(parents=True, exist_ok=True)
    reduced.mkdir(parents=True, exist_ok=True)

    # Custom channel slices.
    left = {"rows": slice(0, 4), "cols": slice(0, 4), "shape": (4, 4)}
    right = {"rows": slice(0, 4), "cols": slice(4, 8), "shape": (4, 4)}

    custom_detector = DetectorConfig(
        left_channel=left,
        right_channel=right,
        clump_positions=[],
        clump_radius=1,
        frame_shape=(4, 8),
    )

    custom_instrument = InstrumentConfig(detector=custom_detector)

    config = ReacherConfig(
        paths=PathConfig(raw_input_dir=str(raw), reduced_output_dir=str(reduced)),
        instrument=custom_instrument,
        processing=ProcessingConfig(overwrite=True),
    )

    # Build a synthetic cube where the left and right halves are distinct.
    cube = np.zeros((2, 4, 8), dtype=np.float32)
    cube[:, :, 0:4] = 11.0
    cube[:, :, 4:8] = 22.0

    in_path = tmp_path / "in.fits"
    fits.writeto(in_path, cube, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"

    ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    ).run()

    left_data = fits.getdata(left_path)
    right_data = fits.getdata(right_path)
    assert left_data.shape == (2, 4, 4)
    assert right_data.shape == (2, 4, 4)
    assert np.allclose(left_data, 11.0)
    assert np.allclose(right_data, 22.0)


def test_bad_pixel_mask_creator_uses_configured_clumps(tmp_path):
    """A custom clump-positions list drives the IRDIS clump mask."""
    from REACHER.steps.bad_pixel import BadPixelMaskCreator

    raw = tmp_path / "raw"
    reduced = tmp_path / "reduced"
    raw.mkdir(parents=True, exist_ok=True)
    reduced.mkdir(parents=True, exist_ok=True)

    custom_detector = DetectorConfig(
        left_channel=IRDIS_LEFT_CHANNEL,
        right_channel=IRDIS_RIGHT_CHANNEL,
        clump_positions=[(5, 5)],  # one clump in a 32x32 frame
        clump_radius=2,
        frame_shape=(32, 32),
    )
    custom_instrument = InstrumentConfig(detector=custom_detector)

    config = ReacherConfig(
        paths=PathConfig(raw_input_dir=str(raw), reduced_output_dir=str(reduced)),
        instrument=custom_instrument,
        processing=ProcessingConfig(overwrite=True),
    )

    # The bad pixel mask step also needs a dark file; build one that
    # matches our custom frame_shape height.
    dark = np.zeros((1024, 32), dtype=np.float32)
    dark_path = tmp_path / "dark.fits"
    fits.writeto(dark_path, dark, overwrite=True)

    bpm_path = tmp_path / "bpm.fits"
    clump_path = tmp_path / "clump.fits"

    BadPixelMaskCreator(
        config=config,
        dark_files=[dark_path],
        bad_pix_mask_path=bpm_path,
        clump_mask_path=clump_path,
    ).run()

    # The clump mask should be of the custom frame shape and have a
    # marked region around (5, 5) -- specifically, the center pixel.
    clump = fits.getdata(clump_path)
    assert clump.shape == (32, 32)
    assert clump[5, 5] == 1.0


# ---------------------------------------------------------------------------
# DPR vocabulary swap
# ---------------------------------------------------------------------------
def test_cube_assembly_filter_uses_configured_science_types(tmp_path):
    """A custom science-types vocabulary drives ``_resolve_science_exp_time``.

    Demonstrates that overriding ``InstrumentConfig.values.science_types``
    causes the batch-level exposure-time filter to pick rows whose
    ``type`` column matches the new vocabulary, not the SPHERE
    ``OBJECT`` default.
    """
    from REACHER.batch.cubes import _resolve_science_exp_time

    raw = tmp_path / "raw"
    reduced = tmp_path / "reduced"
    raw.mkdir(parents=True, exist_ok=True)
    reduced.mkdir(parents=True, exist_ok=True)

    # Custom DPRValues with a non-standard science type.
    custom_values = DPRValues(
        science_types=frozenset({"MY_SCIENCE"}),
        sky_types=frozenset({"SKY"}),
        dark_types=frozenset({"DARK"}),
        flat_types=frozenset({"FLAT"}),
        flux_types=frozenset({"FLUX"}),
        center_types=frozenset({"CENTER"}),
        background_substrings=frozenset({"BG"}),
    )
    custom_instrument = InstrumentConfig(values=custom_values)
    config = ReacherConfig(
        paths=PathConfig(raw_input_dir=str(raw), reduced_output_dir=str(reduced)),
        instrument=custom_instrument,
        processing=ProcessingConfig(overwrite=True, make_plots=False),
    )

    star_df = pd.DataFrame({
        "type": ["MY_SCIENCE", "MY_SCIENCE", "SKY", "OBJECT"],
        "exp_time": [4.0, 4.0, 1.0, 8.0],
    })
    exp_time = _resolve_science_exp_time(star_df, config)
    assert exp_time == 4.0


def test_make_dark_DF_respects_custom_background_substrings(synthetic_raw_layout):
    """The K-band background skip rule honours the configured substrings."""
    from REACHER.discovery.dataframes import make_dark_DF

    # Defaults: SPHERE preset -> all synthetic darks accepted (none
    # are K-band BACKGROUND).
    default_df = make_dark_DF(synthetic_raw_layout["raw"])
    assert not default_df.empty

    # Override background substrings to something not present -- behaviour
    # unchanged (still no skips).
    custom_inst = InstrumentConfig(
        values=DPRValues(
            science_types=frozenset({"OBJECT"}),
            sky_types=frozenset({"SKY"}),
            dark_types=frozenset({"DARK"}),
            flat_types=frozenset({"FLAT,LAMP"}),
            flux_types=frozenset({"OBJECT,FLUX"}),
            center_types=frozenset({"OBJECT,CENTER"}),
            background_substrings=frozenset({"WONT_MATCH_ANYTHING"}),
        )
    )
    other_df = make_dark_DF(
        synthetic_raw_layout["raw"], instrument=custom_inst,
    )
    assert len(other_df) == len(default_df)


def test_default_constructor_no_args_still_works():
    """Smoke test: just constructing the dataclasses with defaults works."""
    config = ReacherConfig(
        paths=PathConfig(
            raw_input_dir="/tmp",
            reduced_output_dir="/tmp",
        ),
    )
    assert isinstance(config.instrument, InstrumentConfig)
    # The factory-style accessors exist.
    assert hasattr(config.instrument, "keys")
    assert hasattr(config.instrument, "values")
    assert hasattr(config.instrument, "detector")
