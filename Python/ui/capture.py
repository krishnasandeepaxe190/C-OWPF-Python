"""OS-level stdout/stderr capture for solver logs.

HiGHS / MOSEK / SCIP / EPANET are C libraries that write their logs (presolve,
primal/dual simplex, branch-and-bound, interior point) straight to file
descriptors 1/2 -- invisible to ``contextlib.redirect_stdout``. These helpers
temporarily point fd 1 and 2 into a temp file so the FULL solver log is
captured, then restore them. Used by the Water, Power and Coupled tabs.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile


def _retarget_log_handlers(new_stream, old_streams):
    """Point plain logging StreamHandlers at ``new_stream`` for the capture.

    CVXPY's MOSEK interface emits the solver log via ``logging`` (not stdout).
    Those handlers hold a reference to the ORIGINAL console stream, which under
    a served app (run_ui.bat) can have an invalid Windows handle -- every MOSEK
    line then dumps an "OSError: [WinError 6]" logging-error traceback. Retarget
    them into the captured stream (so MOSEK logs land in the Solver-log tab) and
    return the originals for restoration. FileHandlers are left untouched.
    """
    changed = []
    names = list(getattr(logging.root.manager, "loggerDict", {}))
    for lg in [logging.getLogger()] + [logging.getLogger(n) for n in names]:
        for h in list(getattr(lg, "handlers", [])):
            if (isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.FileHandler)
                    and getattr(h, "stream", None) in old_streams):
                changed.append((h, h.stream))
                try:
                    h.setStream(new_stream)
                except Exception:
                    h.stream = new_stream
    return changed


@contextlib.contextmanager
def capture_fds():
    """Context manager capturing fd-level stdout+stderr.

    Besides redirecting fd 1/2 into a temp file, ``sys.stdout``/``sys.stderr``
    are REBOUND to fresh UTF-8 writers over the captured descriptors for the
    duration of the block. This matters in a served Streamlit app: the server's
    original ``sys.stdout`` can be a closed/odd console stream (run_ui.bat,
    detached windows, cp1252 consoles), and a bare ``print`` inside the block
    would crash the tab. With the rebind, every print -- ours or a library's --
    lands in the captured log regardless of the host console's state.

    Usage::

        with capture_fds() as cap:
            ...solves / prints...
        text = cap["text"]          # available after the block exits
    """
    fd_out, fd_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    new_stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb"), encoding="utf-8",
                                  errors="replace", line_buffering=True)
    new_stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb"), encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stdout, sys.stderr = new_stdout, new_stderr
    # CVXPY/MOSEK log via `logging`, whose handlers still point at the original
    # (possibly invalid) console stream -- retarget them into the capture, and
    # silence logging's own error dumps for anything we could not retarget.
    old_streams = {old_stdout, old_stderr, sys.__stdout__, sys.__stderr__}
    old_streams.discard(None)
    retargeted = _retarget_log_handlers(new_stderr, old_streams)
    old_raise = logging.raiseExceptions
    logging.raiseExceptions = False
    out = {"text": ""}
    try:
        yield out
    finally:
        for h, s0 in retargeted:
            try:
                h.setStream(s0)
            except Exception:
                h.stream = s0
        logging.raiseExceptions = old_raise
        for s in (new_stdout, new_stderr):
            try:
                s.flush()
                s.close()          # closes only the dup'd descriptor
            except Exception:
                pass
        sys.stdout, sys.stderr = old_stdout, old_stderr
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
