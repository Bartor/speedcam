"""Vehicle speed measurement from nadir drone footage."""

import os

# scipy/MKL load Intel's Fortran runtime, whose console handler aborts the process on Ctrl+C
# ("forrtl: error (200)") before Python can shut down cleanly. Must be set before scipy loads;
# worker processes inherit it.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

# Job worker processes share the server's console, so Ctrl+C reaches them too, possibly while they
# are still importing. The server stops them itself on shutdown, so they ignore it from the start.
import multiprocessing as _mp  # noqa: E402

# (parent_process() is only set after a spawned worker finishes importing; its name is set before)
if _mp.current_process().name != "MainProcess":
    import signal as _signal

    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

__version__ = "0.1.0"
