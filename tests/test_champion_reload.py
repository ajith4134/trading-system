"""LB-09 / RL-030: a bot decides from the champion registered NOW.

`bot_registry` resolved the champion once, inside `segment_bot()`, so the swap
happened at process start and never again. The retrainer refits every four hours
and the bots run 24/7, so **every model fitted between two restarts was
registered and never used** - and nothing said so, because the heartbeat
published the version the bot had LOADED rather than the version that existed.

The user settled the two open questions on 2026-08-19 (RL-030): the swap applies
to NEW ENTRIES ONLY, and the check is a small file read on every poll with the
model loaded only when the id actually changed.

The property worth defending is attribution. A trade opened by one model and
closed by another is attributable to neither - the same failure as crediting one
P&L to all 36 features that could have produced it - so a position already open
keeps the brains that opened it until it closes.
"""
from decimal import Decimal

import pytest

from learn import learned_brains
from segment import live_engine
from segment.live_engine import LiveSegmentEngine


class _Tail:
    """A PROFIT-TAIL that records which positions it was asked to manage."""

    def __init__(self, name):
        self.name = name
        self.managed = []

    def manage(self, *, position, mark, now_ns, frame):     # noqa: ARG002
        self.managed.append(position.symbol)
        from segment.profit_tail import HOLD
        return _Directive(HOLD)


class _Directive:
    def __init__(self, action):
        self.action = action
        self.reason = "HOLD"
        self.evidence = {}
        self.locked_stop = None
        self.reduce_only = False


class _Bot:
    """The smallest thing the engine will treat as a segment bot."""

    def __init__(self, *, model_version="", learned=False, tail=None):
        self.segment = "perp"
        self.venue = "binance-futures"
        self.band = "fast"
        self.model_version = model_version
        self.tail_model_version = ""
        self.learned = learned
        self.profit_tail = tail or _Tail("perp-profit-tail-fast")
        self.bull = _Tail("perp-bull")
        self.bear = _Tail("perp-bear")
        self.extra = {}


def _engine(tmp_path, bot):
    return LiveSegmentEngine(bot=bot, feed=object(), features=object(),
                             root=tmp_path)


@pytest.fixture
def registered(monkeypatch):
    """Control what the registry says is the champion, without a registry."""
    state = {"id": None, "bot": None}

    monkeypatch.setattr(learned_brains, "registered_champion_id",
                        lambda segment, **kw: state["id"])

    def _segment_bot(segment, **kwargs):                    # noqa: ARG001
        if isinstance(state["bot"], Exception):
            raise state["bot"]
        return state["bot"]

    import segment.bot_registry as registry
    monkeypatch.setattr(registry, "segment_bot", _segment_bot)
    return state


def _journalled(engine, outcome):
    """Every decision row of one outcome the engine wrote."""
    import json
    rows = []
    for path in (engine.root / "perp").glob("decisions-*.ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("outcome") == outcome:
                rows.append(row)
    return rows


# --- the swap ---------------------------------------------------------------

def test_a_newly_registered_champion_is_adopted_without_a_restart(tmp_path,
                                                                  registered):
    engine = _engine(tmp_path, _Bot(model_version="old00000", learned=True))
    registered["id"] = "new11111"
    registered["bot"] = _Bot(model_version="new11111", learned=True)

    engine._adopt_new_champion(now_ns=1)

    assert engine.bot.model_version == "new11111"
    swap = _journalled(engine, "CHAMPION_SWAPPED")
    assert len(swap) == 1
    assert swap[0]["evidence"]["from_model"] == "old00000"
    assert swap[0]["evidence"]["to_model"] == "new11111"


def test_an_unchanged_champion_loads_nothing(tmp_path, registered):
    """The check runs every poll, so it must cost a file read and nothing else."""
    engine = _engine(tmp_path, _Bot(model_version="same0000", learned=True))
    registered["id"] = "same0000"
    registered["bot"] = RuntimeError("segment_bot must not be called")

    engine._adopt_new_champion(now_ns=1)

    assert engine.bot.model_version == "same0000"
    assert not _journalled(engine, "CHAMPION_SWAPPED")


def test_no_champion_registered_leaves_the_bot_alone(tmp_path, registered):
    engine = _engine(tmp_path, _Bot(model_version="", learned=False))
    registered["id"] = None

    engine._adopt_new_champion(now_ns=1)

    assert engine.bot.learned is False


def test_a_champion_that_fails_to_load_leaves_the_bot_trading_on_the_old_one(
        tmp_path, registered):
    """Trading on the previous model is correct here; stopping is not."""
    engine = _engine(tmp_path, _Bot(model_version="old00000", learned=True))
    registered["id"] = "new11111"
    registered["bot"] = OSError("registry unreadable")

    engine._adopt_new_champion(now_ns=1)

    assert engine.bot.model_version == "old00000"
    refused = _journalled(engine, "CHAMPION_SWAP_REFUSED")
    assert refused and refused[0]["reason"] == "OSError"


def test_a_champion_refused_by_the_provenance_check_is_not_adopted(tmp_path,
                                                                  registered):
    """`_with_learned_brains` refuses a model fitted on another segment's venues,
    and that refusal must survive the reload path rather than being retried."""
    engine = _engine(tmp_path, _Bot(model_version="old00000", learned=True))
    registered["id"] = "new11111"
    rejected = _Bot(model_version="", learned=False)
    rejected.extra = {"champion_refused": {"why": "fitted on other venues"}}
    registered["bot"] = rejected

    engine._adopt_new_champion(now_ns=1)

    assert engine.bot.model_version == "old00000"
    refused = _journalled(engine, "CHAMPION_SWAP_REFUSED")
    assert refused[0]["reason"] == "NOT_ADOPTED"
    assert refused[0]["evidence"]["champion_refused"]["why"]


# --- attribution: new entries only -----------------------------------------

def test_a_position_open_across_a_swap_is_managed_by_the_brain_that_opened_it(
        tmp_path, registered):
    """A trade opened by one model and closed by another is attributable to
    neither, which is the failure RL-030 settles."""
    from segment.profit_tail import OpenPosition

    old_tail = _Tail("tail-old")
    new_tail = _Tail("tail-new")
    engine = _engine(tmp_path, _Bot(model_version="old00000", learned=True,
                                    tail=old_tail))
    key = ("binance-futures", "BTCUSDT")
    engine.positions[key] = OpenPosition(
        venue=key[0], symbol=key[1], side="LONG", quantity=Decimal("1"),
        entry_price=Decimal("100"), entry_ns=1, hard_stop=Decimal("90"),
        band="fast")
    engine.position_tails[key] = old_tail

    registered["id"] = "new11111"
    registered["bot"] = _Bot(model_version="new11111", learned=True, tail=new_tail)
    engine._adopt_new_champion(now_ns=2)
    engine._manage_open_positions(
        {key: {"bid": Decimal("101"), "ask": Decimal("102"),
               "venue": key[0], "symbol": key[1]}}, now_ns=3)

    assert old_tail.managed == ["BTCUSDT"], "the opening brain must manage it"
    assert new_tail.managed == [], "the new champion must not adopt it"


def test_the_swap_records_how_many_positions_it_left_alone(tmp_path, registered):
    from segment.profit_tail import OpenPosition

    engine = _engine(tmp_path, _Bot(model_version="old00000", learned=True))
    for symbol in ("BTCUSDT", "ETHUSDT"):
        key = ("binance-futures", symbol)
        engine.positions[key] = OpenPosition(
            venue=key[0], symbol=symbol, side="LONG", quantity=Decimal("1"),
            entry_price=Decimal("100"), entry_ns=1, hard_stop=Decimal("90"),
            band="fast")
        engine.position_tails[key] = engine.bot.profit_tail

    registered["id"] = "new11111"
    registered["bot"] = _Bot(model_version="new11111", learned=True)
    engine._adopt_new_champion(now_ns=2)

    swap = _journalled(engine, "CHAMPION_SWAPPED")[0]
    assert swap["evidence"]["open_positions_kept_on_their_own_brains"] == 2
    assert "new entries only" in swap["evidence"]["authority"]


def test_a_position_recovered_after_a_restart_falls_back_to_the_current_brain(
        tmp_path, registered):
    """The object that opened it is gone with the process, and refusing to manage
    a recovered position would leave it open with nobody watching it."""
    from segment.profit_tail import OpenPosition

    current = _Tail("tail-current")
    engine = _engine(tmp_path, _Bot(model_version="m", learned=True, tail=current))
    key = ("binance-futures", "BTCUSDT")
    engine.positions[key] = OpenPosition(
        venue=key[0], symbol=key[1], side="LONG", quantity=Decimal("1"),
        entry_price=Decimal("100"), entry_ns=1, hard_stop=Decimal("90"),
        band="fast")

    engine._manage_open_positions(
        {key: {"bid": Decimal("101"), "ask": Decimal("102"),
               "venue": key[0], "symbol": key[1]}}, now_ns=3)

    assert current.managed == ["BTCUSDT"]


def test_the_registry_read_costs_a_file_and_names_no_champion_when_absent(tmp_path):
    """`registered_champion_id` is called every poll, so it reads the alias file
    and never loads a model."""
    assert learned_brains.registered_champion_id(
        "perp", registry_root=tmp_path) is None

    (tmp_path / "aliases.json").write_text(
        '{"perp-direction-champion": "abc123"}', encoding="utf-8")

    assert learned_brains.registered_champion_id(
        "perp", registry_root=tmp_path) == "abc123"
