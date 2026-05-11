"""Tests for the run-start banner."""
import logging

import pytest

from REACHER.utils import banner as banner_module
from REACHER.utils.banner import print_banner


@pytest.fixture(autouse=True)
def _reset_printed_flag():
    """Reset the once-per-process guard between tests."""
    banner_module._PRINTED = False
    yield
    banner_module._PRINTED = False


def test_print_banner_emits_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="REACHER"):
        print_banner()
    msgs = [r.message for r in caplog.records if r.name == "REACHER"]
    assert any("REACHER v" in m for m in msgs)
    assert any("Run start:" in m for m in msgs)


def test_print_banner_includes_version(caplog):
    from REACHER import __version__
    with caplog.at_level(logging.INFO, logger="REACHER"):
        print_banner()
    msgs = "\n".join(r.message for r in caplog.records if r.name == "REACHER")
    assert __version__ in msgs


def test_print_banner_runs_once_per_process(caplog):
    with caplog.at_level(logging.INFO, logger="REACHER"):
        print_banner()
        first_count = len(caplog.records)
        print_banner()  # no-op
    assert len(caplog.records) == first_count


def test_print_banner_force_re_emits(caplog):
    with caplog.at_level(logging.INFO, logger="REACHER"):
        print_banner()
        first_count = len(caplog.records)
        print_banner(force=True)
    assert len(caplog.records) == 2 * first_count
