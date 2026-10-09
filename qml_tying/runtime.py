"""Cooperative wall-time stopping. Disabled outside the cluster workflow."""
import time


class WorkStopped(RuntimeError):
    """A wall deadline or scheduler signal stopped an incomplete task."""


_deadline = float('inf')
_reason = None


def configure(seconds=None):
    global _deadline, _reason
    _deadline = time.monotonic() + seconds if seconds is not None else float('inf')
    _reason = None


def request_stop(reason='scheduler signal'):
    global _reason
    _reason = str(reason)


def check():
    if _reason is not None:
        raise WorkStopped(_reason)
    if time.monotonic() >= _deadline:
        raise WorkStopped('experiment wall-time allowance reached')


def remaining():
    return max(0., _deadline - time.monotonic())
