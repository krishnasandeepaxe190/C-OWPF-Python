"""OS-level stdout/stderr capture for solver logs.

HiGHS / MOSEK / SCIP / EPANET are C libraries that write their logs (presolve,
primal/dual simplex, branch-and-bound, interior point) straight to file
descriptors 1/2 -- invisible to ``contextlib.redirect_stdout``. These helpers
temporarily point fd 1 and 2 into a temp file so the FULL solver log is
captured, then restore them. Used by the Water, Power and Coupled tabs.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile


@contextlib.contextmanager
def capture_fds():
    """Context manager capturing fd-level stdout+stderr.

    Usage::

        with capture_fds() as cap:
            ...solves...
        text = cap["text"]          # available after the block exits
    """
    fd_out, fd_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    out = {"text": ""}
    try:
        yield out
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(fd_out, 1)
        os.dup2(fd_err, 2)
        os.close(fd_out)
        os.close(fd_err)
        tmp.flush()
        tmp.seek(0)
        out["text"] = tmp.read().decode("utf-8", "ignore")
        tmp.close()


def run_capturing_all(fn):
    """Run ``fn`` capturing fd-level output; returns (fn_result, captured_text)."""
    with capture_fds() as cap:
        result = fn()
    return result, cap["text"]
