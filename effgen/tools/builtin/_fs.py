"""Shared filesystem path-confinement helper for built-in tools.

Document/media tools (ocr, pdf, docx, excel, image_info, qr_read,
audio_transcribe, data_analysis, image_caption, …) take a filesystem path. Left
unconfined, a tool argument is an arbitrary file *read* primitive — and tools
that then upload the bytes to a third-party API (OCR, captioning) turn that into
exfiltration under prompt injection.

Two layers, both built on ``FileOperations``' resolve-then-check model:

* **Default (no ``allowed_directories``):** a *deny-list* blocks the sensitive
  locations an attacker actually wants — ``/etc`` (``/etc/passwd``,
  ``/etc/shadow``), ``/proc``/``/sys``/``/dev``, ``/root``, kernel/boot, mounted
  secrets, and per-user credential stores (``~/.ssh``, ``~/.aws``, ``~/.gnupg``,
  ``~/.kube``, ``~/.config/gcloud``, ``~/.docker``, ``~/.netrc``, …). Ordinary
  files (a report under the project, a scan in a temp dir) still work, so the
  tools stay usable for their intended purpose.
* **Confined (``allowed_directories=[...]``):** the documented opt-in tightens to
  a strict allow-list — only those roots are readable (and the deny-list still
  applies). This is the ``FileOperations`` posture for callers that want it.

Resolution follows symlinks (``Path.resolve``), so a symlink pointing at a denied
location (or outside the allow-list) is rejected too.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PathNotAllowedError(ValueError):
    """Raised when a file path is denied (sensitive location or outside roots).

    Subclasses ValueError so existing tool error handling catches it.
    """


# Absolute directory trees that never hold legitimate tool input and commonly
# hold secrets / kernel interfaces. Reads resolving under these are refused.
_DENY_ABS_DIRS: tuple[str, ...] = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/root",
    "/run/secrets",
    "/var/run/secrets",
)

# Per-user credential stores (resolved against the *owning* user's home as well
# as the current user's home). Relative to a home directory.
_DENY_HOME_SUBPATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".azure",
    ".config/gcloud",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
)


def _denied_roots() -> list[Path]:
    roots = [Path(d) for d in _DENY_ABS_DIRS]
    # Block credential stores under /home/*, /root and the current home, so a
    # path is refused regardless of which user it belongs to.
    home_bases = {Path.home()}
    for base in (Path("/home"), Path("/Users")):
        if base.is_dir():
            try:
                home_bases.update(p for p in base.iterdir() if p.is_dir())
            except OSError:
                # Can't enumerate home roots (perms/race) — safe to skip: the
                # current user's home and /root are already in the deny set, and
                # the absolute deny dirs (/etc, /proc, …) are unaffected.
                pass
    home_bases.add(Path("/root"))
    for base in home_bases:
        for sub in _DENY_HOME_SUBPATHS:
            roots.append(base / sub)
    return roots


def _is_denied(resolved: Path) -> bool:
    for root in _denied_roots():
        try:
            rroot = root.resolve()
        except (OSError, RuntimeError):
            rroot = root
        if resolved == rroot or rroot in resolved.parents:
            return True
    return False


def normalize_allowed_dirs(directories: list[str] | None) -> list[Path] | None:
    """Resolve configured roots, or ``None`` for the default deny-list posture.

    Returning ``None`` (no ``allowed_directories`` given) selects the deny-list
    default; a non-empty list selects strict allow-list confinement.
    """
    if not directories:
        return None
    normalized: list[Path] = []
    for dir_path in directories:
        try:
            path = Path(dir_path).resolve()
        except (OSError, RuntimeError):
            logger.warning("Ignoring invalid allowed directory: %s", dir_path)
            continue
        if path.exists() and path.is_dir():
            normalized.append(path)
        else:
            logger.warning(
                "Allowed directory does not exist or is not a directory: %s", dir_path
            )
    return normalized or None


def is_path_allowed(path: Path, allowed_dirs: list[Path] | None) -> bool:
    """True if ``path`` (resolved) is permitted under the active policy."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    if _is_denied(resolved):
        return False
    if allowed_dirs is None:
        return True  # deny-list mode: anything not denied is allowed
    return any(resolved == root or root in resolved.parents for root in allowed_dirs)


def confine_path(
    path: str,
    allowed_dirs: list[Path] | None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve and validate a file path against the active confinement policy.

    Args:
        path: The user/model-supplied path.
        allowed_dirs: ``None`` for the deny-list default, or roots from
            :func:`normalize_allowed_dirs` for strict allow-list confinement.
        must_exist: When True, also require the file to exist (and be a file).

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        PathNotAllowedError: If the path is denied or outside the roots.
        FileNotFoundError: If ``must_exist`` and the file is missing.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError) as e:
        raise PathNotAllowedError(f"Invalid path '{path}': {e}")

    if not is_path_allowed(resolved, allowed_dirs):
        if allowed_dirs is None:
            raise PathNotAllowedError(
                f"Refusing to read '{path}': it resolves to a protected system "
                "or credential location. Reading secrets via a tool argument is "
                "blocked."
            )
        raise PathNotAllowedError(
            f"Path '{path}' is outside the allowed directories "
            f"({[str(d) for d in allowed_dirs]}). Pass allowed_directories=[…] "
            "to this tool to permit reading from another location."
        )

    if must_exist:
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not resolved.is_file():
            raise PathNotAllowedError(f"Path '{path}' is not a regular file")

    return resolved
