"""Shared helpers for the device tests' error paths.  The normal path of a test that uses them is the same as without
them; only failures change:

  traced(mesh)          a trace capture that is always closed.  On an exception inside the block the capture is ended
                        and the trace released before the exception goes on (a mesh left capturing cannot close its
                        fabric: the next process's ethernet cores stay stuck until the mesh is reset).
  study(run)            a probe's body: an exception is printed and the test passes (a probe's result is its printed
                        lines), except a device timeout, which is raised so that the run ends non-zero and the mesh can
                        be checked (reset) before the next run.
  is_device_timeout(e)  whether an exception is tt-metal's dispatch timeout ("TIMEOUT: device timeout ... potential hang").

    with traced(mesh) as trace:
        outs = run()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
"""

from __future__ import annotations

import contextlib
import os
import traceback

import ttnn

# tt_metal/impl/dispatch/system_memory_manager.cpp: "TIMEOUT: device timeout, potential hang detected, ..." and
# "TIMEOUT: device timeout in fetch queue wait, potential hang detected"
_TIMEOUT_MARKS = ("device timeout", "potential hang detected", "Timeout detected")


def is_device_timeout(e: BaseException) -> bool:
    text = f"{type(e).__name__}: {e}"
    return any(m in text for m in _TIMEOUT_MARKS)


@contextlib.contextmanager
def traced(mesh, cq_id: int = 0):
    """`ttnn.begin_trace_capture` ... `ttnn.end_trace_capture`, yielding the trace id.  On an exception: end the capture
    and release the trace (each step printed if it fails itself), then re-raise the first exception."""
    trace = ttnn.begin_trace_capture(mesh, cq_id=cq_id)
    try:
        yield trace
    except BaseException:
        for step in (
            lambda: ttnn.end_trace_capture(mesh, trace, cq_id=cq_id),
            lambda: ttnn.release_trace(mesh, trace),
        ):
            try:
                step()
            except Exception:  # noqa: BLE001  (the first exception is the one that matters)
                traceback.print_exc()
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=cq_id)


def study(run, strict_env: str | None = None, label: str = "STUDY FAILED (printed, not raised)") -> None:
    """Run a probe's body: an exception is printed with `label`; a device timeout is raised (the run then ends
    non-zero); with `strict_env` set to "1" in the environment every exception is raised."""
    try:
        run()
    except Exception as e:  # noqa: BLE001
        if is_device_timeout(e) or (strict_env and os.environ.get(strict_env) == "1"):
            raise
        traceback.print_exc()
        print(label, flush=True)


__all__ = ["is_device_timeout", "study", "traced"]
