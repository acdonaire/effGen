"""The one git action ``effgen code`` can take: a confirmed commit of its edits.

The coding agent reads a repository through the read-only ``git`` tool. Writing
to one is a separate, deliberately narrow capability that lives here and nowhere
else:

- only ``git add`` and ``git commit``, both limited to the paths the run wrote;
- only after a human confirms, or after the caller passed the flag that means
  "do not ask" — the model can never reach it, because it is not a tool;
- the commit carries the repository's configured identity; an unconfigured
  identity is reported rather than worked around;
- pushing, force-pushing, tagging, amending, resetting, checking out, stashing
  and cleaning are rejected by :func:`ensure_safe`, which every command in this
  module passes through before it runs.

The commit is pathspec-limited (``git commit -- <paths>``), so work the user had
staged for other files stays staged and uncommitted: the agent commits what it
wrote, never what it found.

:func:`unsafe_shell_git` extends the same limits to the shell, so the model
cannot reach ``git push`` or ``git reset --hard`` by asking the ``bash`` tool
instead of using this module.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Git subcommands this module may run. Everything else is rejected.
ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset({
    "add", "commit", "config", "diff", "ls-files", "rev-parse", "status",
})

#: Subcommands that publish, rewrite or discard work. Never run from here.
FORBIDDEN_SUBCOMMANDS: frozenset[str] = frozenset({
    "push", "pull", "fetch", "reset", "clean", "checkout", "switch", "restore",
    "rebase", "merge", "tag", "stash", "cherry-pick", "revert", "rm", "mv",
    "am", "apply", "filter-branch", "update-ref", "gc", "prune", "worktree",
    "submodule", "remote", "branch", "notes", "reflog",
})

#: Flags that would rewrite history, skip the repository's hooks or force an
#: overwrite. Rejected wherever they appear in an argument list.
FORBIDDEN_FLAGS: frozenset[str] = frozenset({
    "--force", "--force-with-lease", "-f", "--amend", "--no-verify", "--hard",
    "--mixed", "--keep", "-D", "--delete",
})

#: ``git`` global options that consume the token after them, so the subcommand
#: is not mistaken for one of their values (``git -C /repo push``).
_GLOBAL_VALUE_OPTIONS: frozenset[str] = frozenset({
    "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path",
    "--config-env", "--super-prefix",
})

# Shell tokens that end one command and begin another, so each `git ...` in a
# chained line is inspected on its own.
_COMMAND_SEPARATORS: frozenset[str] = frozenset({";", "&&", "||", "|", "&", "\n"})

# Fallback scan when a command line cannot be tokenized (unbalanced quotes,
# shell syntax shlex does not model): find any `git <subcommand>` shape.
_GIT_WORD = re.compile(r"(?:^|[\s;&|(])git\s+(?:-\S+\s+)*([a-z][a-z-]*)", re.I)

_TIMEOUT = 30


class UnsafeGitAction(RuntimeError):
    """Raised when a git command outside this module's narrow remit is attempted."""


def ensure_safe(args: list[str]) -> None:
    """Raise :class:`UnsafeGitAction` unless *args* is an allowed git command.

    The first non-flag token is the subcommand and must be in
    :data:`ALLOWED_SUBCOMMANDS`; no option may be one of :data:`FORBIDDEN_FLAGS`.
    Only the options are checked — the commit message and everything after the
    ``--`` pathspec separator are data, not flags. This runs on every command in
    this module, so a future edit cannot widen what a coding session can do to a
    repository without changing the lists above.
    """
    subcommand = next((a for a in args if not a.startswith("-")), "")
    if subcommand in FORBIDDEN_SUBCOMMANDS or subcommand not in ALLOWED_SUBCOMMANDS:
        raise UnsafeGitAction(
            f"git {subcommand or '(none)'} is not available to the coding agent. "
            f"It may only run: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}."
        )
    skip_next = False
    for arg in args:
        if arg == "--":
            break
        if skip_next:
            skip_next = False
            continue
        if arg in ("-m", "--message"):
            skip_next = True
            continue
        if arg in FORBIDDEN_FLAGS:
            raise UnsafeGitAction(
                f"git {subcommand} {arg} is not available to the coding agent: it "
                "would rewrite, force or discard work."
            )


def _forbidden_reason(subcommand: str, options: list[str]) -> str | None:
    """Return why ``git <subcommand> <options>`` may not run, or ``None``."""
    if subcommand in FORBIDDEN_SUBCOMMANDS:
        return (
            f"git {subcommand} is not available to a coding session: it publishes, "
            "rewrites or discards work."
        )
    for option in options:
        if option.split("=", 1)[0] in FORBIDDEN_FLAGS:
            return (
                f"git {subcommand} {option} is not available to a coding session: "
                "it would rewrite, force or discard work."
            )
    return None


def unsafe_shell_git(command: str) -> str | None:
    """Return why *command* may not run as a shell command, or ``None``.

    A coding session reads a repository freely — ``git status``, ``git log``,
    ``git diff`` from the shell are all fine — but it never publishes, rewrites
    or discards work, whichever tool it reaches for. This applies the same
    :data:`FORBIDDEN_SUBCOMMANDS` and :data:`FORBIDDEN_FLAGS` lists
    :func:`ensure_safe` applies to :func:`commit_paths`, so ``bash`` is not a way
    around them. Chained commands are inspected one at a time, and ``git``'s
    global options are skipped so ``git -C other/repo push`` is seen for what it
    is.

    The one repository-changing action a session offers is the confirmed commit
    in this module; a refusal names it.
    """
    text = str(command or "")
    if "git" not in text:
        return None

    try:
        tokens = shlex.split(text, comments=False, posix=True)
    except ValueError:
        tokens = []

    if tokens:
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if Path(token).name != "git":
                continue
            subcommand = ""
            options: list[str] = []
            while index < len(tokens):
                candidate = tokens[index]
                if candidate in _COMMAND_SEPARATORS:
                    break
                index += 1
                if not subcommand:
                    if candidate in _GLOBAL_VALUE_OPTIONS:
                        index += 1
                        continue
                    if candidate.startswith("-"):
                        continue
                    subcommand = candidate
                    continue
                if candidate == "--":
                    break
                if candidate.startswith("-"):
                    options.append(candidate)
            reason = _forbidden_reason(subcommand, options)
            if reason:
                return reason
        return None

    # Untokenizable: fall back to the shape scan rather than letting it through.
    for match in _GIT_WORD.finditer(text):
        reason = _forbidden_reason(match.group(1).lower(), [])
        if reason:
            return reason
    return None


def run_git(root: Path, args: list[str], *, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    """Run a checked-safe git command in *root* and return the completed process.

    Raises:
        UnsafeGitAction: *args* is not one of the commands this module may run.
        RuntimeError: git is not installed.
    """
    ensure_safe(args)
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed, so there is nothing to commit to.")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, allow-listed above
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_identity(root: Path) -> tuple[str | None, str | None]:
    """Return the repository's configured ``(user.name, user.email)``."""
    def _value(key: str) -> str | None:
        try:
            proc = run_git(root, ["config", "--get", key], timeout=10)
        except (UnsafeGitAction, RuntimeError, subprocess.SubprocessError, OSError):
            return None
        value = proc.stdout.strip()
        return value or None

    return _value("user.name"), _value("user.email")


def relative_to_repo(workspace: Path, root: Path, rel_paths: list[str]) -> list[str]:
    """Map workspace-relative paths to repository-relative ones.

    Paths that resolve outside the repository are dropped: a commit only ever
    covers files inside the repository the workspace belongs to.
    """
    out: list[str] = []
    repo_root = Path(root).resolve()
    for rel in rel_paths:
        try:
            resolved = (Path(workspace) / rel).resolve()
            mapped = str(resolved.relative_to(repo_root))
        except (OSError, ValueError):
            continue
        if mapped not in out:
            out.append(mapped)
    return out


def untracked_among(root: Path, paths: list[str]) -> list[str]:
    """Return the *paths* git does not track yet (they would be added by a commit)."""
    if not paths:
        return []
    try:
        proc = run_git(root, ["ls-files", "--", *paths], timeout=15)
    except (UnsafeGitAction, RuntimeError, subprocess.SubprocessError, OSError):
        return []
    tracked = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return [p for p in paths if p not in tracked]


def suggest_message(paths: list[str], task: str = "", new_paths: list[str] | None = None) -> str:
    """Return a plain commit message describing the change.

    The subject names what changed — ``Add`` when every path is new to the
    repository, ``Update`` otherwise — and the task text, when there is one,
    becomes the body so the commit records why. No tool, assistant or vendor name
    appears in it: the commit belongs to the repository's owner.
    """
    names = [Path(p).name for p in paths]
    verb = "Add" if paths and new_paths is not None and set(new_paths) >= set(paths) else "Update"
    if not names:
        subject = "Update workspace files"
    elif len(names) == 1:
        subject = f"{verb} {names[0]}"
    elif len(names) == 2:
        subject = f"{verb} {names[0]} and {names[1]}"
    else:
        subject = f"{verb} {names[0]} and {len(names) - 1} more files"
    if len(subject) > 72:
        subject = subject[:69] + "..."

    body = " ".join((task or "").split())
    if not body:
        return subject
    if len(body) > 500:
        body = body[:497] + "..."
    return f"{subject}\n\n{body}"


@dataclass
class CommitOutcome:
    """The result of a commit attempt."""

    success: bool
    message: str
    detail: str = ""
    commit: str | None = None
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome as a JSON-serializable dict."""
        return {
            "success": self.success,
            "commit": self.commit,
            "message": self.message,
            "detail": self.detail,
            "paths": list(self.paths),
        }


def other_staged_paths(root: Path, paths: list[str]) -> list[str]:
    """Return staged paths that are *not* part of *paths*.

    Reported before a commit so the user knows work they staged themselves is
    present — the commit will not include it, and this says so.
    """
    try:
        proc = run_git(root, ["diff", "--cached", "--name-only"], timeout=15)
    except (UnsafeGitAction, RuntimeError, subprocess.SubprocessError, OSError):
        return []
    staged = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    known = set(paths)
    return [p for p in staged if p not in known]


def commit_paths(root: Path, paths: list[str], message: str) -> CommitOutcome:
    """Stage *paths* and commit exactly those paths with *message*.

    The commit is limited to *paths*, so anything else in the index stays there.
    The repository's configured identity is used as-is; it is never overridden.

    Returns a :class:`CommitOutcome` describing what happened — a failed commit
    reports git's own message rather than raising.
    """
    root = Path(root)
    if not paths:
        return CommitOutcome(False, message, "No files to commit.")
    if not message.strip():
        return CommitOutcome(False, message, "A commit message is required.")

    name, email = git_identity(root)
    if not name or not email:
        return CommitOutcome(
            False, message,
            "This repository has no commit identity configured. Set one with "
            "'git config user.name \"...\"' and 'git config user.email \"...\"', "
            "then commit again.",
        )

    try:
        staged = run_git(root, ["add", "--", *paths])
        if staged.returncode != 0:
            return CommitOutcome(
                False, message, (staged.stderr or staged.stdout).strip() or "git add failed."
            )
        done = run_git(root, ["commit", "-m", message, "--", *paths])
    except UnsafeGitAction as exc:  # pragma: no cover - defensive
        return CommitOutcome(False, message, str(exc))
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        return CommitOutcome(False, message, str(exc))

    if done.returncode != 0:
        return CommitOutcome(
            False, message,
            (done.stderr or done.stdout).strip() or "git commit failed.", paths=paths,
        )

    sha = ""
    try:
        rev = run_git(root, ["rev-parse", "--short", "HEAD"], timeout=10)
        sha = rev.stdout.strip()
    except (UnsafeGitAction, RuntimeError, subprocess.SubprocessError, OSError):
        sha = ""
    return CommitOutcome(True, message, "", commit=sha or None, paths=list(paths))
