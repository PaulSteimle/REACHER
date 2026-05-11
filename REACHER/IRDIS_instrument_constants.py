"""Module-level SPHERE/IRDIS instrument constants.

These constants are the **SPHERE/IRDIS source-of-truth** values that
:meth:`REACHER.config.InstrumentConfig.sphere_irdis` consumes when it
builds the default SPHERE preset. They are retained at module scope
(rather than being moved entirely onto the dataclass) for two reasons:

1. Backwards compatibility: existing call sites and tests import
   ``IRDIS_LEFT_CHANNEL`` / ``IRDIS_RIGHT_CHANNEL`` / ``SPHERE_PA_OFFSET``
   / ... directly from this module. Keeping them here preserves that API.
2. Single source of truth: the SPHERE factory pulls every value from
   here, so updating an instrument constant (e.g. a new true-north
   correction) only requires editing this file.

Citations
---------
- IRDIS detector pixel ranges and clump artifact positions: SPHERE
  Manual + instrument team private communication. The LEFT and RIGHT
  channels have DIFFERENT row ranges by detector geometry (not a bug).
- SPHERE_PA_OFFSET = 135.99 deg: SPHERE Manual.
- SPHERE_TRUE_NORTH_CORRECTION = -1.75 deg: Maire et al. 2020,
  A&A 633, A63.
- SPHERE_ANAMORPHIC_RATIO = (1.0000, 1.0062): A. Vigan, private
  communication. The y-axis scaling differs from the x-axis scaling by
  ~0.6%; the anamorphic correction therefore requires independent x/y
  rescaling -- a vip-fft transform cannot express it, so the cube
  assembly step falls back to opencv + lanczos4 for this stage.
"""

# IRDIS detector geometry. LEFT and RIGHT row slices are intentionally
# different: the right channel is offset by ~10 px in y due to the
# physical detector layout. Both channels output (995, 875) frames.
IRDIS_LEFT_CHANNEL = {"rows": slice(25, 1020), "cols": slice(50, 925), "shape": (995, 875)}
IRDIS_RIGHT_CHANNEL = {"rows": slice(15, 1010), "cols": slice(1080, 1955), "shape": (995, 875)}

# IRDIS detector clump artifacts: persistent hot regions on the
# detector. Always masked + interpolated regardless of dark-frame
# statistics.
IRDIS_CLUMP_POSITIONS = [(368, 394), (362, 394), (356, 394), (311, 1400)]
IRDIS_CLUMP_RADIUS = 10

# Raw IRDIS frame shape (rows, cols).
IRDIS_FRAME_SHAPE = (1024, 2048)

# SPHERE optics constants (used to convert telescope angle metadata
# into on-sky parallactic angles).
SPHERE_PA_OFFSET = 135.99
SPHERE_TRUE_NORTH_CORRECTION = -1.75
SPHERE_ANAMORPHIC_RATIO = (1.0000, 1.0062)

# SPHERE FITS header keyword names. The non-standard keywords use the
# ``HIERARCH ESO ...`` convention; the FITS-standard keywords
# (``EXPTIME``, ``DATE-OBS``, ``OBJECT``) are not stored here because
# they are universal across telescopes.
SPHERE_HEADER_FILTER_NAME = "HIERARCH ESO INS COMB IFLT"
SPHERE_HEADER_ND_FILTER_NAME = "HIERARCH ESO INS4 FILT2 NAME"
SPHERE_HEADER_ND_FILTER_ID = "HIERARCH ESO INS4 FILT2 ID"
SPHERE_HEADER_DPR_TYPE = "HIERARCH ESO DPR TYPE"
SPHERE_HEADER_DETECTOR_TEMP = "HIERARCH ESO INS4 TEMP449 VAL"
SPHERE_HEADER_PARANG_START = "HIERARCH ESO TEL PARANG START"
SPHERE_HEADER_PARANG_END = "HIERARCH ESO TEL PARANG END"
SPHERE_HEADER_DITHER_X = "HIERARCH ESO INS1 DITH POSX"
SPHERE_HEADER_DITHER_Y = "HIERARCH ESO INS1 DITH POSY"

# SPHERE DPR-type vocabulary. Each set lists the ``HIERARCH ESO DPR
# TYPE`` values that classify a frame as that pipeline concept. Sets
# (rather than single strings) allow multi-token vocabularies for
# instruments like NACO that use both ``SCIENCE`` and ``IMAGE,OBJECT``
# for science frames.
SPHERE_DPR_SCIENCE_TYPES = frozenset({"OBJECT"})
SPHERE_DPR_SKY_TYPES = frozenset({"SKY"})
SPHERE_DPR_DARK_TYPES = frozenset({"DARK"})
SPHERE_DPR_FLAT_TYPES = frozenset({"FLAT,LAMP"})
SPHERE_DPR_FLUX_TYPES = frozenset({"OBJECT,FLUX"})
SPHERE_DPR_CENTER_TYPES = frozenset({"OBJECT,CENTER"})

# Substring-matched DPR-type tokens. Used by the dataset-discovery
# layer to filter out K-band BACKGROUND frames that live under the
# DARK/ directory: their DPR type contains ``BACKGROUND`` somewhere in
# the (potentially composite) string. The match is a substring rather
# than equality because the legacy data has variants like
# ``DARK,BACKGROUND``.
SPHERE_DPR_BACKGROUND_SUBSTRINGS = frozenset({"BACKGROUND"})
