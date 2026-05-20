"""Display formatting helpers for log output."""
from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Return a human-readable string for a duration in seconds.

    Resolution narrows as magnitude grows:

    * ``< 60 s``    -> ``"5.83s"`` (2 decimals)
    * ``< 1 h``     -> ``"2m 5.0s"`` (1 decimal on seconds)
    * ``>= 1 h``    -> ``"1h 2m 5s"`` (whole seconds)

    Negative inputs are treated as zero.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}m {secs:.1f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds - hours * 3600 - minutes * 60))
    # Handle 59.5 -> 60 rollover from the round().
    if secs == 60:
        secs = 0
        minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
    return f"{hours}h {minutes}m {secs}s"
