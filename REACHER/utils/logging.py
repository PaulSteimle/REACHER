import logging, sys, os
import multiprocessing


def _in_worker_process() -> bool:
    """True iff called from a multiprocessing.Pool worker (not the main process).

    Uses ``current_process().name`` rather than ``parent_process()`` because
    the latter still returns None during the worker's top-level entry-script
    re-import under ``spawn`` semantics; ``current_process().name`` is set
    to ``'SpawnPoolWorker-N'`` (or similar) at that point.
    """
    return multiprocessing.current_process().name != "MainProcess"


def log_blank_line() -> None:
    """Emit a bare ``\\n`` to every root-logger handler's stream.

    Bypasses the formatter (so no timestamp/level prefix) and the
    ``PrintLogger`` shim installed by :func:`setup_logging` (which
    strips empty messages). Used by :class:`REACHER.pipeline.Pipeline`
    to visually separate per-step output blocks.
    """
    for handler in logging.getLogger().handlers:
        stream = getattr(handler, "stream", None)
        if stream is None:
            continue
        try:
            stream.write("\n")
            stream.flush()
        except Exception:
            pass


def setup_logging(log_file="log.log", level=logging.INFO):
    class PrintLogger:
        def __init__(self, l=logging.INFO):
            self.l = l
            self.logger = logging.getLogger()
            self._orig = sys.__stdout__ if l == logging.INFO else sys.__stderr__
        def write(self, msg):
            msg = msg.strip()
            if msg: self.logger.log(self.l, msg)
        def flush(self): pass
        def fileno(self): return self._orig.fileno()

    logger = logging.getLogger()
    logger.setLevel(level)

    # In a multiprocessing.Pool worker: the parent already owns the log
    # file and the banner. Workers must NOT re-truncate the file or
    # re-print the banner. We still attach a StreamHandler so per-step
    # INFO lines (e.g. "Calibrating X | dark=Y flat=Z") reach the
    # inherited stderr.
    if _in_worker_process():
        if not logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    sys.stdout = PrintLogger(logging.INFO)
    sys.stderr = PrintLogger(logging.ERROR)

    # Once-per-process run-start banner (version + git metadata).
    from .banner import print_banner
    print_banner()