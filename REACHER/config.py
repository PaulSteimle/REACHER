"""Configuration dataclasses for the REACHER pipeline.

The composed :class:`ReacherConfig` aggregates four focused
sub-configurations:

* :class:`PathConfig`         -- input / output directories.
* :class:`InstrumentConfig`   -- instrument-coupling profile
  (header keywords, DPR vocabulary, detector geometry, plus the
  optical scalars used by the cube-assembly step).
* :class:`ProcessingConfig`   -- runtime flags (overwrite, n_proc, ...).
* :class:`CalibrationConfig`  -- calibration parameters.

Retargeting the pipeline to a different telescope/instrument is done
by swapping the :class:`InstrumentConfig` profile. The default profile
returned by :meth:`InstrumentConfig.sphere_irdis` is the SPHERE/IRDIS
preset; constructing :class:`InstrumentConfig` with no arguments
returns the same SPHERE preset for back-compatibility.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from .exceptions import InvalidConfigError
from .IRDIS_instrument_constants import (
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


@dataclass
class PathConfig:
    """Paths for input/output directories."""
    raw_input_dir: str = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_raw_files/'
    reduced_output_dir: str = '/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_reduced_cubes/'

    def __post_init__(self):
        if not Path(self.raw_input_dir).exists():
            raise InvalidConfigError(f"raw_input_dir does not exist: {self.raw_input_dir}")
        if not Path(self.reduced_output_dir).exists():
            Path(self.reduced_output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Instrument profile sub-dataclasses
# ---------------------------------------------------------------------------
# These three frozen dataclasses define what is "instrument-coupling": the
# FITS header keyword names, the DPR-type vocabulary, and the detector
# geometry. They are bundled into :class:`InstrumentConfig` so that the
# entire instrument profile can be swapped in one place. Frozen so the
# config object is safe to share across process boundaries (multiprocessing
# Pool workers receive a pickled copy).
#
# FITS-standard keywords (EXPTIME, DATE-OBS, OBJECT, DATE) are deliberately
# NOT modelled here: they are universal across telescopes and live in
# the FITS standard, so callers can rely on them directly.

@dataclass(frozen=True)
class HeaderKeywords:
    """Instrument-specific FITS header keyword names.

    The keyword names used to look up instrument-specific metadata in
    a FITS header. Defaults are the SPHERE/IRDIS ``HIERARCH ESO ...``
    keywords; non-SPHERE profiles set their own keywords.

    Parameters
    ----------
    filter_name
        Combined IRDIS filter name (e.g. ``H``, ``DB_H23``).
    nd_filter_name
        ND filter human-readable name (e.g. ``ND_1.0``).
    nd_filter_id
        ND filter machine ID (column selector for the ND profile file).
    dpr_type
        Data Product Reduction type. The value classifies the frame
        as ``OBJECT`` / ``SKY`` / ``DARK`` / etc.
    detector_temp
        Detector temperature, used for sky-subtraction scaling.
    parang_start
        Parallactic angle at the start of the exposure (degrees).
    parang_end
        Parallactic angle at the end of the exposure (degrees).
    dither_x, dither_y
        Telescope dither offsets (pixels) applied by ``np.roll`` in the
        cube-assembly step.

    Notes
    -----
    FITS-standard keywords such as ``EXPTIME``, ``DATE-OBS``, ``DATE``
    and ``OBJECT`` are not modelled here -- they are universal across
    telescopes and are read directly by their standard names.
    """

    filter_name: str
    nd_filter_name: str
    nd_filter_id: str
    dpr_type: str
    detector_temp: str
    parang_start: str
    parang_end: str
    dither_x: str
    dither_y: str


@dataclass(frozen=True)
class DPRValues:
    """Instrument-specific DPR-type vocabulary.

    Each field is a :class:`frozenset` of header strings classifying
    a frame as that pipeline concept. Sets (rather than single
    strings) support multi-token vocabularies for instruments where
    e.g. both ``SCIENCE`` and ``IMAGE,OBJECT`` mark science frames.

    Parameters
    ----------
    science_types
        DPR-type strings that mark a frame as a science (target) frame.
    sky_types
        DPR-type strings that mark a frame as a sky / background frame.
    dark_types
        DPR-type strings for raw dark calibrations.
    flat_types
        DPR-type strings for raw flat-field calibrations.
    flux_types
        DPR-type strings for flux-calibration target frames.
    center_types
        DPR-type strings for satellite-spot CENTER frames.
    background_substrings
        Substrings whose presence in a DPR-type string marks a frame as
        a background / lamp variant. Used by the dataset-discovery
        layer to drop K-band ``DARK,BACKGROUND`` rows that live under
        the ``DARK/`` directory.  Substring matching is deliberate:
        the legacy data has composite tokens (e.g. ``DARK,BACKGROUND``)
        that must match.
    """

    science_types: FrozenSet[str]
    sky_types: FrozenSet[str]
    dark_types: FrozenSet[str]
    flat_types: FrozenSet[str]
    flux_types: FrozenSet[str]
    center_types: FrozenSet[str]
    background_substrings: FrozenSet[str]


@dataclass(frozen=True)
class DetectorConfig:
    """Instrument-specific detector geometry.

    Parameters
    ----------
    left_channel, right_channel
        Mapping with keys ``rows`` (``slice``), ``cols`` (``slice``)
        and ``shape`` (``tuple``). For SPHERE/IRDIS the two channels
        have intentionally different row slices -- this is detector
        geometry, not a bug.
    clump_positions
        Sequence of ``(row, col)`` positions of persistent detector
        artifacts ("clumps") that are always masked + interpolated
        regardless of the per-night dark statistics.
    clump_radius
        Radius (pixels) of each circular clump mask.
    frame_shape
        Raw detector frame shape ``(rows, cols)``.

    Notes
    -----
    ``left_channel`` / ``right_channel`` are mappings (not slice
    objects) so they can carry the LEFT/RIGHT pre-split output shape
    in the ``shape`` key. Tests / step code access them as
    ``detector.left_channel["rows"]`` etc.
    """

    left_channel: Dict
    right_channel: Dict
    clump_positions: List[Tuple[int, int]]
    clump_radius: int
    frame_shape: Tuple[int, int]


# ---------------------------------------------------------------------------
# Default factory helpers (module-level so the dataclass default_factory
# closures stay simple).
# ---------------------------------------------------------------------------
def _sphere_header_keywords() -> HeaderKeywords:
    return HeaderKeywords(
        filter_name=SPHERE_HEADER_FILTER_NAME,
        nd_filter_name=SPHERE_HEADER_ND_FILTER_NAME,
        nd_filter_id=SPHERE_HEADER_ND_FILTER_ID,
        dpr_type=SPHERE_HEADER_DPR_TYPE,
        detector_temp=SPHERE_HEADER_DETECTOR_TEMP,
        parang_start=SPHERE_HEADER_PARANG_START,
        parang_end=SPHERE_HEADER_PARANG_END,
        dither_x=SPHERE_HEADER_DITHER_X,
        dither_y=SPHERE_HEADER_DITHER_Y,
    )


def _sphere_dpr_values() -> DPRValues:
    return DPRValues(
        science_types=SPHERE_DPR_SCIENCE_TYPES,
        sky_types=SPHERE_DPR_SKY_TYPES,
        dark_types=SPHERE_DPR_DARK_TYPES,
        flat_types=SPHERE_DPR_FLAT_TYPES,
        flux_types=SPHERE_DPR_FLUX_TYPES,
        center_types=SPHERE_DPR_CENTER_TYPES,
        background_substrings=SPHERE_DPR_BACKGROUND_SUBSTRINGS,
    )


def _sphere_detector_config() -> DetectorConfig:
    return DetectorConfig(
        left_channel=dict(IRDIS_LEFT_CHANNEL),
        right_channel=dict(IRDIS_RIGHT_CHANNEL),
        clump_positions=list(IRDIS_CLUMP_POSITIONS),
        clump_radius=IRDIS_CLUMP_RADIUS,
        frame_shape=IRDIS_FRAME_SHAPE,
    )


@dataclass
class InstrumentConfig:
    """Typed instrument coupling profile.

    Bundles the instrument-specific optical scalars, FITS header
    keywords, DPR vocabulary, and detector geometry into a single
    swappable object. Constructing :class:`InstrumentConfig` with no
    arguments yields the SPHERE/IRDIS preset, matching the legacy
    behaviour.

    To retarget the pipeline to a different instrument, build a custom
    profile (preferably via a new ``cls.<instr>()`` classmethod
    alongside :meth:`sphere_irdis`).

    Parameters
    ----------
    sphere_pa_offset
        Position-angle offset applied to per-frame parallactic angles
        in the cube assembly step (degrees). SPHERE default: 135.99.
    sphere_true_north_correction
        True-north correction added to the PA (degrees). SPHERE
        default: -1.75 (Maire et al. 2020, A&A 633, A63).
    sphere_anamorphic_ratio
        ``(x_scale, y_scale)`` anamorphic rescaling. SPHERE default:
        ``(1.0000, 1.0062)`` (A. Vigan, priv. comm.). The x/y axes
        scale independently; the cube-assembly step falls back to
        opencv + lanczos4 because vip-fft cannot express asymmetric
        axis scaling.
    keys
        :class:`HeaderKeywords` profile.
    values
        :class:`DPRValues` profile (DPR-type vocabularies).
    detector
        :class:`DetectorConfig` profile (channel slices, clump mask).

    Notes
    -----
    The two ``sphere_*`` scalar names are kept for back-compatibility
    with existing code paths (cube_assembly reads
    ``config.instrument.sphere_pa_offset``). When defining a non-SPHERE
    profile, set these scalars to the appropriate values for that
    instrument; the names are historical, not semantic.
    """

    sphere_pa_offset: float = SPHERE_PA_OFFSET
    sphere_true_north_correction: float = SPHERE_TRUE_NORTH_CORRECTION
    sphere_anamorphic_ratio: Tuple[float, float] = SPHERE_ANAMORPHIC_RATIO
    keys: HeaderKeywords = field(default_factory=_sphere_header_keywords)
    values: DPRValues = field(default_factory=_sphere_dpr_values)
    detector: DetectorConfig = field(default_factory=_sphere_detector_config)

    @classmethod
    def sphere_irdis(cls) -> "InstrumentConfig":
        """Return the SPHERE/IRDIS instrument profile.

        Equivalent to ``InstrumentConfig()``. Provided as a named
        factory so the retargeting story is self-documenting:

        .. code-block:: python

            config = ReacherConfig(
                instrument=InstrumentConfig.sphere_irdis(),
            )

        Returns
        -------
        InstrumentConfig
            A fresh SPHERE/IRDIS profile.
        """
        return cls(
            sphere_pa_offset=SPHERE_PA_OFFSET,
            sphere_true_north_correction=SPHERE_TRUE_NORTH_CORRECTION,
            sphere_anamorphic_ratio=SPHERE_ANAMORPHIC_RATIO,
            keys=_sphere_header_keywords(),
            values=_sphere_dpr_values(),
            detector=_sphere_detector_config(),
        )


@dataclass
class ProcessingConfig:
    """Processing parameters."""
    nproc: int = 4
    overwrite: bool = True
    make_plots: bool = True
    skip_bad_pix: bool = False
    skip_cosmics: bool = False
    do_flux: bool = True
    do_cubes: bool = True
    do_angles: bool = True
    only_angles: bool = False
    only_flux: bool = False
    skip_dither_correction: bool = False
    subtract_median_science_sky_bkg: bool = False
    correct_anamorphism: bool = True
    full_output: bool = False
    save_fullsize_flux: bool = False
    box_radius: int = 2
    n_sigma: int = 9


@dataclass
class CalibrationConfig:
    """Calibration parameters."""
    calib_validity_threshold: int = 8
    star_loc: Tuple[int, int] = (401, 466)
    star_center_index: int = 0
    image_shift_library: str = 'vip-fft'
    image_rotation_library: str = 'vip-fft'
    image_shift_interpolation: str = 'bicubic'
    image_rotation_interpolation: str = 'bicubic'
    flux_cut_out_size: int = 41


@dataclass
class ReacherConfig:
    """Master configuration combining all sub-configs."""
    paths: Optional[PathConfig] = None
    instrument: Optional[InstrumentConfig] = None
    processing: Optional[ProcessingConfig] = None
    calibration: Optional[CalibrationConfig] = None

    def __post_init__(self):
        if self.paths is None:
            self.paths = PathConfig()
        if self.instrument is None:
            self.instrument = InstrumentConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.calibration is None:
            self.calibration = CalibrationConfig()
