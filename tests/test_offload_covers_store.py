"""The offload covers the store, not only the raw tape.

`store/funding` is polled straight into the store with no raw counterpart -
`~/capture/raw/binance-funding` does not exist - so a raw-only backup silently
omits the one dataset whose loss cannot be undone: the OBSERVED funding record
whose start date is the clock every promotion waits on. Measured 2026-08-17,
the bucket held raw/, ledger/ and universe/ and nothing else.

`funding_reconstructed` cannot stand in for it. Its availability times are the
fetch, which is deliberately what makes it safe for research and useless for a
backtest - so a restore from it would silently reset the honest record.

The second property here matters as much as the first. The store is written
live: `.writing-part-*.parquet` files exist while a dataset is being appended,
and copying one uploads a truncated parquet that reads as complete. That is
the same defect the raw path guards with its `.writing` markers, and this
script's own header calls it out - so the store copy must exclude them.
"""
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offload_to_gcs.sh"


def test_offload_script_copies_the_store_as_well_as_raw():
    body = SCRIPT.read_text()
    assert "store" in body, (
        "offload_to_gcs.sh must copy $CAPTURE_ROOT/store; funding lives only there")


def test_offload_script_still_copies_raw():
    body = SCRIPT.read_text()
    assert "raw" in body, "the raw tape backup must not be lost while adding the store"


def test_the_store_is_copied_by_the_extras_loop_rather_than_a_second_mechanism():
    """One copy path for the small trees, so there is one thing to keep right."""
    body = SCRIPT.read_text()
    extras = [line for line in body.splitlines() if line.strip().startswith("for extra in")]
    assert extras, "the extras loop is where ledger and universe are copied"
    assert "store" in extras[0], (
        f"store belongs in the extras loop; found {extras[0]!r}")


def test_a_live_parquet_being_written_is_excluded_from_the_copy():
    """A truncated parquet that reads as complete is worse than no copy."""
    body = SCRIPT.read_text()
    assert "writing-part" in body, (
        "the store copy must exclude .writing-part-*.parquet; uploading one "
        "lands a truncated file that reads as a complete dataset")
