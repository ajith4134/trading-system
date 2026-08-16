"""A model store keyed by content hash, with aliases that move and a history that does not.

`FEATURES.md` §3 (P1): *"Model registry with aliases — MLflow aliases; registry
**stages** are deprecated"*. Ledger MD-012, whose note reads *"MLflow, file
store, no daemon"*.

## Built rather than adopted, and the lesson is adopted without the dependency

`DECISIONS.md` §10 puts *"ML/DL research pipeline + validation gates"* and *"the
experiment ledger"* on the **build** side, and NautilusTrader is the only thing
on the adopt side. MLflow with a file store is ~50 MB of dependency and a
database layer for what is, at the size this project needs, a directory of files
and a JSON pointer file.

What IS adopted is the lesson MLflow learned the expensive way: **aliases, not
stages.** MLflow's `Staging`/`Production`/`Archived` vocabulary is deprecated
because a fixed set of stage names is somebody else's workflow — it cannot
express two models in production at once, cannot name a shadow, and turns every
project-specific concept into an abuse of one of three words. An alias is a
string a caller chooses, and a version can carry as many as it needs.

Three things a file store gives us that MLflow does not, and that this project
has written requirements for:

* **Content-hash identity** (ledger MD-029: *"third-party model weights treated
  as untrusted binaries, pin by content hash"*). The version id IS the SHA-256 of
  the bytes, so registering the same model twice is the same version, and a
  changed file cannot keep an old id.
* **Verification on read.** `load` re-hashes and refuses on mismatch. MD-029
  aims that at third-party weights; it is applied to our own too, because a
  corrupted file on disk is indistinguishable from a hostile one at the moment it
  is loaded, and only one of those is unlikely.
* **No deserialisation.** The registry stores and returns `bytes`. It never
  unpickles, so loading a model cannot execute code, and the caller's serialiser
  is the caller's business.

## Aliases move; the record of where they pointed does not

`assign_alias("production", version)` repoints a name. Every assignment appends
to `alias-history.ndjson`, which is never rewritten.

That file is the answer to the only question that matters after a bad fill:
**what was live when this trade happened.** A pointer that silently moves cannot
answer it, and the reconciliation policy this project has written down depends on
being able to. A registry that only knows its current state is a registry that
can only ever describe now.

## A version without a trial is refused

Every registration names the `trial_id` from `validation.trial_registry` that
produced it. Refused rather than optional: a model whose trial was never counted
is a model whose N nobody knows, and every promotion gate downstream divides its
protection by N. The failure is silent in exactly the wrong direction — the model
still loads, still predicts, and its significance is computed against a search
that looks smaller than it was.

## What this deliberately does not do

No promotion logic, no gates, no "is this good enough". `validation.
promotion_pipeline` owns that and already refuses things. A registry that decided
what may be promoted would be a second place for the promotion rules to live, and
the second place is the one that drifts.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODELS_DIR = "models"
ALIASES_FILE = "aliases.json"
ALIAS_HISTORY_FILE = "alias-history.ndjson"
METADATA_SUFFIX = ".json"
ARTIFACT_SUFFIX = ".bin"

# Long enough that a collision is not a thing anyone needs to reason about, short
# enough to read in a log line. The FULL hash is stored in the metadata and is
# what `load` verifies against - this is a display and lookup key, never the
# integrity check.
VERSION_ID_CHARS = 16


class TrialRequired(ValueError):
    """A model was registered without the trial that produced it.

    Refused rather than allowed with a null. A model whose trial was never
    counted is a model whose N nobody knows, and every promotion gate divides its
    protection by N - so the omission weakens the gates silently, in the
    direction that promotes things.
    """


class UnknownAlias(KeyError):
    """No such alias. Raised rather than returning None.

    A caller resolving `production` and getting `None` has to remember to check;
    the one that forgets loads nothing and trades on whatever it had. Every
    consumer of this registry is a pricing or sizing path.
    """


class UnknownVersion(KeyError):
    """No such version in this registry."""


class ArtifactCorrupt(RuntimeError):
    """The bytes on disk do not hash to the id they are filed under.

    MD-029's rule, applied to our own artefacts: at the moment of loading, a
    corrupted file and a substituted one are indistinguishable, and only one of
    them is unlikely.
    """


@dataclass(frozen=True)
class ModelVersion:
    """One registered artefact and everything needed to audit it later.

    `metrics` is stored verbatim rather than in a fixed schema: what is worth
    recording differs by learner, and a schema here would be a third place - after
    the trial registry and the axis verdicts - where the same numbers have to be
    kept in agreement.
    """
    version_id: str
    sha256: str
    trial_id: int
    family: str
    name: str
    created_at_ns: int
    size_bytes: int
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "version_id": self.version_id, "sha256": self.sha256,
            "trial_id": self.trial_id, "family": self.family, "name": self.name,
            "created_at_ns": self.created_at_ns, "size_bytes": self.size_bytes,
            "metrics": self.metrics, "notes": self.notes,
        }


@dataclass(frozen=True)
class AliasAssignment:
    """One movement of an alias, as it was appended."""
    alias: str
    version_id: str
    previous_version_id: str | None
    assigned_at_ns: int
    reason: str


class ModelRegistry:
    """A directory of artefacts, a pointer file, and an append-only history."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._models = self._root / MODELS_DIR

    # --- writing ----------------------------------------------------------

    def register(self, artifact: bytes, *, trial_id: int | None, family: str,
                 name: str, metrics: dict[str, Any] | None = None,
                 notes: str = "") -> ModelVersion:
        """Store `artifact` under its own content hash and return the version.

        Registering identical bytes twice returns the SAME version rather than a
        second one. That is what content-addressing buys: a retrain that produced
        exactly the previous model is visible as such, instead of appearing as a
        new version whose only difference is a timestamp. The existing metadata
        wins - the first registration's trial is the one that produced these
        bytes, and overwriting it would attribute them to a later look at the
        data.
        """
        if trial_id is None:
            raise TrialRequired(
                f"model {name!r} was registered with no trial_id. Refused: a "
                f"model whose trial was never counted is a model whose N nobody "
                f"knows, and every promotion gate divides its protection by N")
        if not artifact:
            raise ValueError(
                f"model {name!r} has no bytes. An empty artefact loads, hashes "
                f"and verifies perfectly, and predicts nothing")

        digest = hashlib.sha256(artifact).hexdigest()
        version_id = digest[:VERSION_ID_CHARS]
        self._models.mkdir(parents=True, exist_ok=True)
        metadata_path = self._models / f"{version_id}{METADATA_SUFFIX}"
        if metadata_path.is_file():
            return self.version(version_id)

        version = ModelVersion(
            version_id=version_id, sha256=digest, trial_id=int(trial_id),
            family=family, name=name, created_at_ns=time.time_ns(),
            size_bytes=len(artifact), metrics=dict(metrics or {}), notes=notes)
        # Artefact first, then metadata. A crash between the two leaves an
        # unreferenced blob, which is inert; the reverse order leaves metadata
        # pointing at bytes that are not there, which is a version that resolves
        # and cannot load.
        _write_atomic(self._models / f"{version_id}{ARTIFACT_SUFFIX}", artifact)
        _write_atomic(metadata_path,
                      json.dumps(version.as_dict(), indent=1).encode("utf-8"))
        return version

    def assign_alias(self, alias: str, version_id: str, *,
                     reason: str = "") -> AliasAssignment:
        """Point `alias` at `version_id`, and append where it pointed before.

        The history append is not a convenience. After a bad fill the question is
        what was live at the time, and a pointer that only knows its current
        value cannot answer it.
        """
        if not alias:
            raise ValueError("an alias must have a name")
        version = self.version(version_id)          # raises if unknown
        aliases = self._read_aliases()
        previous = aliases.get(alias)

        assignment = AliasAssignment(
            alias=alias, version_id=version.version_id,
            previous_version_id=previous, assigned_at_ns=time.time_ns(),
            reason=reason)
        aliases[alias] = version.version_id
        _write_atomic(self._root / ALIASES_FILE,
                      json.dumps(aliases, indent=1, sort_keys=True).encode("utf-8"))
        self._append_history(assignment)
        return assignment

    def _append_history(self, assignment: AliasAssignment) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "alias": assignment.alias, "version_id": assignment.version_id,
            "previous_version_id": assignment.previous_version_id,
            "assigned_at_ns": assignment.assigned_at_ns,
            "reason": assignment.reason,
        }) + "\n"
        path = self._root / ALIAS_HISTORY_FILE
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    # --- reading ----------------------------------------------------------

    def version(self, version_id: str) -> ModelVersion:
        path = self._models / f"{version_id}{METADATA_SUFFIX}"
        if not path.is_file():
            raise UnknownVersion(
                f"no version {version_id!r} in {self._root}")
        return ModelVersion(**json.loads(path.read_text(encoding="utf-8")))

    def versions(self) -> list[ModelVersion]:
        """Every version, oldest registration first."""
        if not self._models.is_dir():
            return []
        found = [self.version(p.stem)
                 for p in sorted(self._models.glob(f"*{METADATA_SUFFIX}"))]
        return sorted(found, key=lambda v: v.created_at_ns)

    def aliases(self) -> dict[str, str]:
        return dict(self._read_aliases())

    def resolve(self, alias: str) -> ModelVersion:
        """The version an alias points at, or `UnknownAlias`."""
        aliases = self._read_aliases()
        if alias not in aliases:
            raise UnknownAlias(
                f"no alias {alias!r} in {self._root}. Known: "
                f"{sorted(aliases) or 'none'}")
        return self.version(aliases[alias])

    def aliases_of(self, version_id: str) -> list[str]:
        """Every alias pointing at this version.

        A version can carry several - `production` and `champion` on the same
        bytes is a normal state, and the fixed-vocabulary stage model this
        replaces could not express it.
        """
        return sorted(a for a, v in self._read_aliases().items()
                      if v == version_id)

    def alias_history(self, alias: str | None = None) -> list[AliasAssignment]:
        """Every assignment ever made, oldest first, optionally for one alias."""
        path = self._root / ALIAS_HISTORY_FILE
        if not path.is_file():
            return []
        out: list[AliasAssignment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only the last line can be torn - a process killed mid-append.
                # Skipped rather than fatal: a truncated final record must not
                # make the whole history unreadable.
                continue
            if alias is not None and row["alias"] != alias:
                continue
            out.append(AliasAssignment(**row))
        return out

    def load(self, version_id: str) -> bytes:
        """The artefact's bytes, verified against the hash they are filed under.

        Never deserialised here. The registry returns `bytes` and the caller's
        serialiser is the caller's business, so loading a model cannot execute
        code - which matters most for exactly the third-party weights MD-029 is
        about, and costs nothing for our own.
        """
        version = self.version(version_id)
        path = self._models / f"{version_id}{ARTIFACT_SUFFIX}"
        if not path.is_file():
            raise UnknownVersion(
                f"version {version_id!r} has metadata but no artefact at {path}")
        artifact = path.read_bytes()
        digest = hashlib.sha256(artifact).hexdigest()
        if digest != version.sha256:
            raise ArtifactCorrupt(
                f"version {version_id!r} hashes to {digest[:VERSION_ID_CHARS]} "
                f"but is filed under {version.sha256[:VERSION_ID_CHARS]}. "
                f"Refusing to load: at this moment a corrupted file and a "
                f"substituted one are indistinguishable")
        return artifact

    def load_alias(self, alias: str) -> tuple[ModelVersion, bytes]:
        """Resolve and load in one step, which is what every caller wants."""
        version = self.resolve(alias)
        return version, self.load(version.version_id)

    def _read_aliases(self) -> dict[str, str]:
        path = self._root / ALIASES_FILE
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write through a temporary file and rename.

    A half-written metadata file is a version that resolves and cannot load, and
    a half-written alias file is a pointer to nothing. `rename` within a
    directory is atomic on POSIX, so a reader sees the old file or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
