"""Order-intent write-ahead log — EX-001, EX-003, and the Phase 0 WAL requirement.

`DECISIONS.md` §6 / `FEATURES.md` §9: *"Write-ahead order-intent log with startup
reconciliation."* `FEATURES.md` §5, EX-001: *"Client order ID derived
deterministically; query by original ID on timeout, **never blind-retry**."*

The ledger's citation for why: **Everbright Securities lost roughly $3.8 billion**
doing the opposite. Blind-retrying an order whose response was lost is not a
recovery strategy; it is a second order.

Two mechanisms, and they only work together:

**Write before send.** The intent is on disk and fsynced *before* anything is
transmitted. If the process dies between write and send, recovery sees an intent with
unknown fate and can ask the venue. If it died between send and write - which is what
happens when the log is written afterwards - recovery has no idea an order exists, and
the position is invisible.

**Deterministic client order id.** Same intent, same id, always. The id is what makes
asking the venue possible at all: on a timeout you query by the original id rather
than sending again. A random or timestamped id turns every uncertain send into a
guaranteed duplicate on retry.

`EX-003` rides along, because an expired intent is the other way to get an order
nobody wants: *"a signal computed 5 minutes ago must not fire now."*

**Nothing here talks to a venue.** This is durability and recovery only - no
credentials, no network, no order placement. `submit` takes a transport callable, so
tests drive it with doubles and the live path is a separate decision.
"""
import json
from decimal import Decimal

import pytest

from execution.order_intent_wal import (
    IntentExpired,
    OrderIntent,
    OrderIntentWal,
    WalCorrupt,
    derive_client_order_id,
)

NOW = 1_800_000_000_000_000_000
SECOND = 1_000_000_000


def an_intent(symbol="BTCUSDT", side="BUY", quantity="0.5", created_at_ns=NOW,
              strategy="momentum-v1", **kw):
    return OrderIntent(strategy=strategy, symbol=symbol, venue="binance",
                       side=side, quantity=Decimal(quantity),
                       created_at_ns=created_at_ns,
                       valid_for_ns=30 * SECOND, **kw)


# --- deterministic ids ------------------------------------------------------

def test_the_same_intent_always_derives_the_same_id():
    """The property the whole recovery story rests on. Without it, "query by the
    original id" has no original id to query by."""
    assert derive_client_order_id(an_intent()) == derive_client_order_id(an_intent())


def test_a_different_quantity_derives_a_different_id():
    assert derive_client_order_id(an_intent(quantity="0.5")) != \
           derive_client_order_id(an_intent(quantity="0.6"))


def test_a_different_side_derives_a_different_id():
    assert derive_client_order_id(an_intent(side="BUY")) != \
           derive_client_order_id(an_intent(side="SELL"))


def test_a_different_creation_time_derives_a_different_id():
    """Otherwise two genuinely separate decisions to buy the same size collapse into
    one id, and the second is silently treated as a duplicate of the first."""
    assert derive_client_order_id(an_intent(created_at_ns=NOW)) != \
           derive_client_order_id(an_intent(created_at_ns=NOW + SECOND))


def test_the_id_fits_the_venue_length_limit():
    """Binance caps newClientOrderId at 36 characters. An id the venue truncates is
    an id you cannot query by, which defeats the point."""
    assert len(derive_client_order_id(an_intent())) <= 36


def test_the_id_is_venue_safe_characters_only():
    identifier = derive_client_order_id(an_intent(symbol="1000SHIBUSDT"))
    assert identifier.replace("-", "").replace("_", "").isalnum()


# --- write before send ------------------------------------------------------

def test_the_intent_is_durable_before_the_transport_is_called(tmp_path):
    """**The ordering that matters.** If the log is written after sending, a crash in
    between leaves an order at the venue that nothing local knows about."""
    wal = OrderIntentWal(tmp_path)
    seen_on_disk = {}

    def transport(intent, client_order_id):
        seen_on_disk["records"] = len(OrderIntentWal(tmp_path).records())
        return {"status": "accepted"}

    wal.submit(an_intent(), transport, now_ns=NOW)
    assert seen_on_disk["records"] >= 1, (
        "the transport ran before the intent was durable")


def test_a_transport_failure_leaves_the_intent_recorded_as_unknown(tmp_path):
    """A timeout is not a rejection. The venue may have accepted it, so the intent
    must survive as unresolved - the one state that prompts a query rather than a
    resend."""
    wal = OrderIntentWal(tmp_path)

    def times_out(intent, client_order_id):
        raise TimeoutError("no response")

    with pytest.raises(TimeoutError):
        wal.submit(an_intent(), times_out, now_ns=NOW)

    unresolved = OrderIntentWal(tmp_path).unresolved()
    assert len(unresolved) == 1
    assert unresolved[0]["outcome"] == "unknown"
    assert "TimeoutError" in unresolved[0]["error"]


def test_an_accepted_intent_is_not_unresolved(tmp_path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"status": "accepted"}, now_ns=NOW)
    assert OrderIntentWal(tmp_path).unresolved() == []


def test_the_log_survives_a_restart(tmp_path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"status": "accepted"}, now_ns=NOW)
    assert len(OrderIntentWal(tmp_path).records()) >= 1


def test_the_log_is_append_only(tmp_path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(created_at_ns=NOW), lambda i, c: {"ok": 1}, now_ns=NOW)
    before = (tmp_path / "order_intents.ndjson").read_text(encoding="utf-8")
    wal.submit(an_intent(created_at_ns=NOW + SECOND), lambda i, c: {"ok": 1},
               now_ns=NOW + SECOND)
    after = (tmp_path / "order_intents.ndjson").read_text(encoding="utf-8")
    assert after.startswith(before)


# --- never blind-retry ------------------------------------------------------

def test_resubmitting_the_same_intent_is_refused(tmp_path):
    """The Everbright failure, blocked. The same intent resubmitted after an
    uncertain outcome must not reach the transport a second time - it must be
    resolved by querying the venue with the id already recorded."""
    wal = OrderIntentWal(tmp_path)
    calls = []

    def times_out(intent, client_order_id):
        calls.append(client_order_id)
        raise TimeoutError("no response")

    with pytest.raises(TimeoutError):
        wal.submit(an_intent(), times_out, now_ns=NOW)
    with pytest.raises(RuntimeError, match="already"):
        wal.submit(an_intent(), times_out, now_ns=NOW + SECOND)

    assert len(calls) == 1, "the intent was blind-retried"


def test_resolving_an_unknown_intent_closes_it(tmp_path):
    """The correct recovery: ask the venue by the recorded id, then record what it
    said. This is what makes the refusal above safe rather than a dead end."""
    wal = OrderIntentWal(tmp_path)
    with pytest.raises(TimeoutError):
        wal.submit(an_intent(), lambda i, c: (_ for _ in ()).throw(TimeoutError()),
                   now_ns=NOW)

    identifier = wal.unresolved()[0]["client_order_id"]
    wal.resolve(identifier, outcome="accepted",
                detail="venue reported the order live on query")
    assert OrderIntentWal(tmp_path).unresolved() == []


def test_resolving_an_unknown_id_is_refused(tmp_path):
    with pytest.raises(KeyError):
        OrderIntentWal(tmp_path).resolve("never-seen", outcome="accepted",
                                         detail="")


# --- signal expiry ----------------------------------------------------------

def test_an_expired_intent_is_refused_before_being_written(tmp_path):
    """EX-003: "a signal computed 5 minutes ago must not fire now." Refused before
    the log write, because an intent that must never be sent should not leave a
    record implying it might have been."""
    wal = OrderIntentWal(tmp_path)
    with pytest.raises(IntentExpired):
        wal.submit(an_intent(created_at_ns=NOW), lambda i, c: {"ok": 1},
                   now_ns=NOW + 31 * SECOND)
    assert wal.records() == []


def test_an_intent_inside_its_validity_window_is_accepted(tmp_path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(created_at_ns=NOW), lambda i, c: {"ok": 1},
               now_ns=NOW + 29 * SECOND)
    assert len(wal.records()) == 1


def test_expiry_is_recorded_even_though_the_order_never_went(tmp_path):
    """Abandoned-and-logged, per `FEATURES.md` §3b. An expiry that leaves no trace
    hides that the system is consistently too slow to act on its own signals."""
    wal = OrderIntentWal(tmp_path)
    with pytest.raises(IntentExpired):
        wal.submit(an_intent(created_at_ns=NOW), lambda i, c: {"ok": 1},
                   now_ns=NOW + 31 * SECOND)
    assert [e["event"] for e in wal.expiries()] == ["expired"]


def test_a_non_positive_validity_window_is_refused():
    with pytest.raises(ValueError):
        OrderIntent(strategy="s", symbol="S", venue="v", side="BUY",
                    quantity=Decimal("1"), created_at_ns=NOW, valid_for_ns=0)


# --- fail closed on damage --------------------------------------------------

def test_a_corrupt_log_raises_rather_than_reporting_an_empty_history(tmp_path):
    """An empty history means "no orders outstanding", which is the most dangerous
    possible misreading of an unreadable WAL."""
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"ok": 1}, now_ns=NOW)
    (tmp_path / "order_intents.ndjson").write_text("{ broken\n", encoding="utf-8")
    with pytest.raises(WalCorrupt):
        OrderIntentWal(tmp_path).records()


def test_a_torn_final_line_is_tolerated_and_treated_as_unresolved(tmp_path):
    """A process killed mid-append leaves a partial line. That intent may have
    reached the venue, so it has to surface as unresolved rather than be dropped."""
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"ok": 1}, now_ns=NOW)
    with (tmp_path / "order_intents.ndjson").open("a", encoding="utf-8") as fh:
        fh.write('{"event": "submitted", "client_order_id": "half-writ')

    reloaded = OrderIntentWal(tmp_path)
    assert reloaded.has_torn_tail is True
    assert any(r["outcome"] == "unknown" for r in reloaded.unresolved())


def test_the_written_record_names_the_strategy_and_the_quantity(tmp_path):
    """Auditability. An order nobody can attribute to a strategy cannot be
    attributed in the ledger either, and per-feature attribution being wrong is the
    documented cause of the -$837 result."""
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(strategy="carry-v2", quantity="1.25"),
               lambda i, c: {"ok": 1}, now_ns=NOW)
    record = wal.records()[0]
    assert record["strategy"] == "carry-v2"
    assert record["quantity"] == "1.25"
    assert json.dumps(record)          # serialisable, no Decimal leakage
