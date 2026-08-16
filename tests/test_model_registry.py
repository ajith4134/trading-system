"""Aliases move; the record of where they pointed does not.

The test that carries the row is `test_the_history_answers_what_was_live_at_a
_past_moment`. After a bad fill the only question is what model was live when the
trade happened, and a registry that knows only its current state cannot answer
it - which is exactly what a pointer file alone would be.

The rest defend identity: bytes are the version, a changed file cannot keep an
old id, and a model with no counted trial cannot be registered at all.
"""
import json

import pytest

from models.model_registry import (
    ALIAS_HISTORY_FILE,
    ArtifactCorrupt,
    ModelRegistry,
    TrialRequired,
    UnknownAlias,
    UnknownVersion,
)


def _registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def _register(registry, artifact=b"model-bytes", trial_id=1, name="gbt",
              **kwargs):
    return registry.register(artifact, trial_id=trial_id, family="carry",
                             name=name, **kwargs)


# --- identity is the content ---------------------------------------------

def test_the_same_bytes_are_the_same_version(tmp_path):
    """A retrain that reproduced the previous model is visible as such, rather
    than as a new version whose only difference is a timestamp."""
    registry = _registry(tmp_path)
    first = _register(registry, b"identical", trial_id=1)
    second = _register(registry, b"identical", trial_id=2)

    assert first.version_id == second.version_id
    assert len(registry.versions()) == 1


def test_the_first_registration_keeps_the_trial_that_produced_the_bytes(tmp_path):
    """Overwriting with the later trial would attribute these bytes to a later
    look at the data than the one that actually made them."""
    registry = _registry(tmp_path)
    _register(registry, b"identical", trial_id=1)
    again = _register(registry, b"identical", trial_id=99)

    assert again.trial_id == 1


def test_different_bytes_are_different_versions(tmp_path):
    registry = _registry(tmp_path)
    first = _register(registry, b"one")
    second = _register(registry, b"two")

    assert first.version_id != second.version_id
    assert len(registry.versions()) == 2


def test_a_model_with_no_counted_trial_is_refused(tmp_path):
    """Its N is unknown, and every promotion gate divides its protection by N -
    so the omission weakens the gates silently, in the promoting direction."""
    with pytest.raises(TrialRequired):
        _registry(tmp_path).register(b"orphan", trial_id=None, family="carry",
                                     name="orphan")


def test_an_empty_artefact_is_refused(tmp_path):
    """It loads, hashes and verifies perfectly, and predicts nothing."""
    with pytest.raises(ValueError, match="predicts nothing"):
        _register(_registry(tmp_path), b"")


# --- verification on read -------------------------------------------------

def test_the_bytes_are_verified_against_the_hash_they_are_filed_under(tmp_path):
    """MD-029's rule applied to our own artefacts: at the moment of loading, a
    corrupted file and a substituted one are indistinguishable, and only one of
    them is unlikely."""
    registry = _registry(tmp_path)
    version = _register(registry, b"trustworthy")
    artefact = tmp_path / "registry" / "models" / f"{version.version_id}.bin"
    artefact.write_bytes(b"tampered-with")

    with pytest.raises(ArtifactCorrupt):
        registry.load(version.version_id)


def test_a_good_artefact_round_trips_unchanged(tmp_path):
    """The verification must not be the only thing that works."""
    registry = _registry(tmp_path)
    version = _register(registry, b"\x00\x01binary\xff")

    assert registry.load(version.version_id) == b"\x00\x01binary\xff"


def test_metadata_without_an_artefact_is_a_named_failure(tmp_path):
    registry = _registry(tmp_path)
    version = _register(registry)
    (tmp_path / "registry" / "models" / f"{version.version_id}.bin").unlink()

    with pytest.raises(UnknownVersion, match="no artefact"):
        registry.load(version.version_id)


# --- aliases, not stages --------------------------------------------------

def test_an_alias_can_be_repointed(tmp_path):
    registry = _registry(tmp_path)
    first = _register(registry, b"v1")
    second = _register(registry, b"v2")

    registry.assign_alias("production", first.version_id)
    assert registry.resolve("production").version_id == first.version_id

    registry.assign_alias("production", second.version_id)
    assert registry.resolve("production").version_id == second.version_id


def test_one_version_can_carry_several_aliases(tmp_path):
    """`production` and `champion` on the same bytes is a normal state, and the
    fixed-vocabulary stage model this replaces could not express it."""
    registry = _registry(tmp_path)
    version = _register(registry)
    registry.assign_alias("production", version.version_id)
    registry.assign_alias("champion", version.version_id)

    assert registry.aliases_of(version.version_id) == ["champion", "production"]


def test_alias_names_are_the_caller_s_own(tmp_path):
    """No fixed vocabulary. A shadow, a per-venue model, a per-brain one - the
    reason MLflow deprecated stages is that three words are somebody else's
    workflow."""
    registry = _registry(tmp_path)
    version = _register(registry)
    for alias in ("shadow-binance", "brain-1-bull", "rollback-target"):
        registry.assign_alias(alias, version.version_id)

    assert set(registry.aliases()) == {"shadow-binance", "brain-1-bull",
                                       "rollback-target"}


def test_resolving_an_unknown_alias_raises_rather_than_returning_none(tmp_path):
    """A caller that forgets the None check loads nothing and trades on whatever
    it had. Every consumer of this registry is a pricing or sizing path."""
    with pytest.raises(UnknownAlias):
        _registry(tmp_path).resolve("production")


def test_an_alias_cannot_point_at_a_version_that_does_not_exist(tmp_path):
    with pytest.raises(UnknownVersion):
        _registry(tmp_path).assign_alias("production", "deadbeefdeadbeef")


# --- the history ----------------------------------------------------------

def test_the_history_answers_what_was_live_at_a_past_moment(tmp_path):
    """The row's whole point. A pointer file knows only now."""
    registry = _registry(tmp_path)
    first = _register(registry, b"v1")
    second = _register(registry, b"v2")
    third = _register(registry, b"v3")

    registry.assign_alias("production", first.version_id, reason="first live")
    registry.assign_alias("production", second.version_id, reason="retrain")
    registry.assign_alias("production", third.version_id, reason="retrain")

    history = registry.alias_history("production")
    assert [h.version_id for h in history] == [
        first.version_id, second.version_id, third.version_id]
    assert [h.previous_version_id for h in history] == [
        None, first.version_id, second.version_id]
    assert history[1].reason == "retrain"


def test_the_history_is_append_only(tmp_path):
    """Rewritten, it would be a second copy of the current state - and the state
    is already in the pointer file."""
    registry = _registry(tmp_path)
    version = _register(registry)
    registry.assign_alias("production", version.version_id)
    path = tmp_path / "registry" / ALIAS_HISTORY_FILE
    after_first = path.read_text(encoding="utf-8")

    registry.assign_alias("production", version.version_id)

    assert path.read_text(encoding="utf-8").startswith(after_first)


def test_a_torn_final_history_line_does_not_hide_the_rest(tmp_path):
    """A process killed mid-append must not make the whole history unreadable -
    it is the one file that answers a question nothing else can."""
    registry = _registry(tmp_path)
    version = _register(registry)
    registry.assign_alias("production", version.version_id)
    path = tmp_path / "registry" / ALIAS_HISTORY_FILE
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"alias": "produ')

    assert len(registry.alias_history()) == 1


def test_the_history_survives_a_reopened_registry(tmp_path):
    """It is on disk, not in memory - which is the difference between a record
    and a session."""
    version_id = _register(_registry(tmp_path)).version_id
    _registry(tmp_path).assign_alias("production", version_id)

    assert len(_registry(tmp_path).alias_history("production")) == 1


# --- metadata -------------------------------------------------------------

def test_metrics_are_stored_verbatim(tmp_path):
    """No fixed schema: what is worth recording differs by learner, and a schema
    here would be a third place - after the trial registry and the axis verdicts -
    where the same numbers have to be kept in agreement."""
    registry = _registry(tmp_path)
    version = _register(registry, metrics={"accuracy": 0.61, "base_rate": 0.52,
                                           "sharpe": None})

    assert registry.version(version.version_id).metrics["sharpe"] is None
    assert registry.version(version.version_id).metrics["accuracy"] == 0.61


def test_the_metadata_file_is_readable_without_this_module(tmp_path):
    """A registry only its own code can read is a registry nobody can audit
    after the code changes."""
    registry = _registry(tmp_path)
    version = _register(registry, name="lightgbm-carry")
    path = tmp_path / "registry" / "models" / f"{version.version_id}.json"

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["name"] == "lightgbm-carry"
    assert stored["sha256"].startswith(version.version_id)


def test_load_alias_resolves_and_verifies_in_one_step(tmp_path):
    registry = _registry(tmp_path)
    version = _register(registry, b"payload")
    registry.assign_alias("production", version.version_id)

    resolved, artifact = registry.load_alias("production")

    assert resolved.version_id == version.version_id
    assert artifact == b"payload"


def test_an_empty_registry_reports_emptiness_rather_than_raising(tmp_path):
    registry = _registry(tmp_path)

    assert registry.versions() == []
    assert registry.aliases() == {}
    assert registry.alias_history() == []
