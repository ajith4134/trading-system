"""Make LightGBM importable on a box with no system OpenMP runtime.

LightGBM's shared library links against `libgomp.so.1`, the GNU OpenMP runtime.
This machine does not have it on the loader path and cannot get it the usual way:
there is no passwordless `sudo`, so `apt-get install libgomp1` is not available.
Importing LightGBM therefore fails with

    OSError: libgomp.so.1: cannot open shared object file: No such file or directory

which reads like a broken install and is not one — the wheel is fine and one
system library is missing.

## What this does

`ensure_openmp()` tries the plain import first and returns immediately if it
works. That is the common case on any normal machine and the case that must stay
free: a box with a working system libgomp should never touch the fallback, and
the fallback must never shadow a runtime the system already chose.

Only if the import raises does it look for a copy in `SEARCH_PATHS` and load it
with `RTLD_GLOBAL`, which publishes the symbols into the process so the
subsequent LightGBM load resolves against them. It is not a `LD_LIBRARY_PATH`
change, so nothing about the environment of any other process is altered — which
matters here, because the supervisors that will eventually run training are
started by a shell script and an environment variable set in one of them would be
a fifth place for this to be configured.

## Where the copy came from, and how to replace it

`~/.local/lib/libgomp.so.1`, copied 2026-08-16 from the conda-forge
`libgomp-16.1.0` package already present in this box's micromamba cache. To
recreate it from scratch:

    micromamba create -p ~/.local/openmp -c conda-forge libgomp
    cp -L ~/.local/openmp/lib/libgomp.so.1 ~/.local/lib/libgomp.so.1

The mamba cache paths are searched too, after the owned copy, so a fresh clone of
this repo on this box works before anyone runs that command.

## Why not avoid the dependency instead

`DECISIONS.md` §8 settles it: *"LightGBM is the default. Every neural component
must beat LightGBM and a linear baseline on our own data before it ships."*
Swapping in a pure-Python gradient booster to dodge one missing `.so` would
change a decided thing to avoid a fifteen-line fix, and scikit-learn's
`HistGradientBoosting` needs the same runtime anyway.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

_LIBRARY = "libgomp.so.1"

# Searched in order. The owned copy first, because it is the one this project is
# responsible for; the package caches after it, so a fresh checkout works before
# anyone has run the copy command in the docstring.
SEARCH_PATHS: tuple[Path, ...] = (
    Path.home() / ".local" / "lib" / _LIBRARY,
    Path.home() / ".local" / "share" / "mamba" / "lib" / _LIBRARY,
    Path.home() / ".local" / "gitenv" / "lib" / _LIBRARY,
)

# Set this to a path to override the search entirely - for a machine where the
# runtime lives somewhere none of the above would look.
PATH_ENV_VAR = "TRADING_SYSTEM_LIBGOMP"

_loaded: ctypes.CDLL | None = None


class OpenMpRuntimeMissing(RuntimeError):
    """No OpenMP runtime could be found, and LightGBM cannot import without one.

    Carries the paths that were searched and the command that fixes it. A bare
    "cannot open shared object file" sends a reader to look for a broken wheel,
    which is the wrong place: the wheel is fine and one system library is absent.
    """


def ensure_openmp() -> None:
    """Load an OpenMP runtime if LightGBM cannot import without one.

    Idempotent, and a no-op wherever the system already provides the runtime.
    Called at the top of every module that imports LightGBM rather than once at
    package import, so that importing `models` for something unrelated - the
    naive baseline, say - does not drag in a native library it has no use for.
    """
    global _loaded
    try:
        import lightgbm  # noqa: F401 - imported for the side effect of loading
        return
    except OSError as error:
        if _LIBRARY not in str(error):
            raise

    override = os.environ.get(PATH_ENV_VAR)
    candidates = ([Path(override)] if override else []) + list(SEARCH_PATHS)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        # RTLD_GLOBAL so the symbols are visible to the LightGBM library loaded
        # afterwards. Without it the load succeeds and resolves nothing.
        _loaded = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        return

    raise OpenMpRuntimeMissing(
        f"LightGBM needs {_LIBRARY} and this machine has no system copy. "
        f"Searched: {', '.join(str(c) for c in candidates)}. Fix with\n"
        f"    micromamba create -p ~/.local/openmp -c conda-forge libgomp\n"
        f"    cp -L ~/.local/openmp/lib/{_LIBRARY} ~/.local/lib/{_LIBRARY}\n"
        f"or point {PATH_ENV_VAR} at an existing copy. Refusing rather than "
        f"falling back to a different learner: DECISIONS.md section 8 settles "
        f"LightGBM as the default, and silently training something else would "
        f"make every comparison against it meaningless")
