"""A board that reports its own age, because a timestamp is not a warning.

On 2026-08-10 12:53 the boards generator stopped writing and the box went off.
The server kept serving, the tunnel kept resolving, and every page rendered a
complete, plausible, entirely green board — five days out of date, with its
generation timestamp printed at the top the whole time. Nobody noticed, because
a timestamp is something you have to read and then do arithmetic on.

The measurement has to happen in the READER's browser. A server-rendered
"generated 5 days ago" is impossible for the only case that matters: the server
that would render it is the one that stopped.

These tests check the contract of what is emitted, since the arithmetic itself
runs in a browser: that the fallback shown when script does not run is honest,
that the threshold and the instant travel with the page, and that an unparseable
stamp lands on AGE UNKNOWN rather than on freshness.
"""
from statuswall.staleness_banner import STALE_AFTER_SECONDS, render_staleness_banner


def test_the_generation_instant_travels_with_the_page():
    html = render_staleness_banner(1_786_800_000, "2026-08-15 17:00 UTC")
    assert 'data-generated="1786800000"' in html


def test_the_threshold_travels_with_the_page_rather_than_being_baked_in():
    html = render_staleness_banner(1_786_800_000, "x", stale_after_seconds=42)
    assert 'data-stale-after="42"' in html


def test_the_default_threshold_is_three_missed_regenerations():
    # The generator rebuilds every 300s; past three misses something is wrong
    # rather than slow.
    assert STALE_AFTER_SECONDS == 900


def test_a_page_whose_script_never_runs_says_the_age_is_unknown():
    """The server-rendered fallback. It must not be the healthy styling — a
    board that cannot measure itself is in the NOT MEASURED state, which Rule 8
    says renders as its own thing and never as green."""
    html = render_staleness_banner(1_786_800_000, "2026-08-15 17:00 UTC")
    assert "AGE UNKNOWN" in html
    assert "staleness-unknown" in html


def test_the_fallback_still_names_when_the_page_was_generated():
    html = render_staleness_banner(1_786_800_000, "2026-08-15 17:00 UTC")
    assert "2026-08-15 17:00 UTC" in html


def test_the_stale_message_names_the_generator_not_just_the_age():
    """'This board is 5 days old' invites a reload. 'The generator has stopped
    writing it' names the thing to go and fix — which is what nobody worked out
    for five days."""
    html = render_staleness_banner(1_786_800_000, "x")
    assert "generator has stopped writing it" in html


def test_the_stale_message_says_the_page_is_not_the_current_state():
    html = render_staleness_banner(1_786_800_000, "x")
    assert "not " in html and "current state of the system" in html


def test_the_banner_repaints_itself_so_an_open_tab_goes_red_on_its_own():
    html = render_staleness_banner(1_786_800_000, "x")
    assert "setInterval" in html


def test_the_label_is_escaped_so_a_page_title_cannot_inject_markup():
    html = render_staleness_banner(1, "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_nothing_is_fetched_from_anywhere():
    """A staleness check that needed a CDN would go blank in exactly the network
    conditions that produce a stale board."""
    html = render_staleness_banner(1_786_800_000, "x")
    for forbidden in ("http://", "https://", "fetch(", "XMLHttpRequest", "src="):
        assert forbidden not in html


def test_a_wall_whose_timestamp_cannot_be_parsed_falls_back_to_unknown():
    """`_epoch_of` returns 0 for an unreadable stamp, and 0 makes the banner
    render AGE UNKNOWN rather than claiming freshness — the safe direction."""
    from statuswall.wall_page import _epoch_of
    assert _epoch_of("not a timestamp") == 0
    assert _epoch_of("") == 0


def test_a_wall_timestamp_in_the_normal_format_parses():
    from statuswall.wall_page import _epoch_of
    # Cross-checked against `date -u -d @1786813200`, not against the function
    # under test — a constant taken from the code it verifies proves nothing.
    assert _epoch_of("2026-08-15T17:00:00Z") == 1_786_813_200
