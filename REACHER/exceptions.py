"""Custom exceptions for the REACHER pipeline."""

class ReacherError(Exception):
    pass
class CalibrationFileNotFoundError(ReacherError):
    pass
class ShapeMismatchError(ReacherError):
    pass
class FITSReadError(ReacherError):
    pass
class FITSWriteError(ReacherError):
    pass
class StepExecutionError(ReacherError):
    pass
class HeaderKeyError(ReacherError):
    pass
class InvalidConfigError(ReacherError):
    pass
class DataFrameError(ReacherError):
    pass
