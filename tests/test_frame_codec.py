from capture.frame_codec import (
    escape_payload, unescape_payload, IndexEntry,
    encode_index_entry, decode_index_entry,
)


def test_clean_payload_is_untouched():
    payload = '{"e":"depthUpdate","U":1,"u":2}'
    escaped, was_escaped = escape_payload(payload)
    assert escaped == payload
    assert was_escaped is False


def test_newline_is_escaped_and_roundtrips():
    payload = '{"a":"x\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_carriage_return_is_escaped_and_roundtrips():
    payload = '{"a":"x\r\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\r" not in escaped and "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_backslash_roundtrips_without_false_escape():
    payload = r'{"a":"C:\path"}'
    escaped, was_escaped = escape_payload(payload)
    assert unescape_payload(escaped) == payload


def test_backslash_and_newline_in_one_payload_roundtrip():
    r"""Both in ONE payload, which is the only shape that constrains the escaping.

    Nothing in the suite pinned backslash escaping: the backslash test used a
    payload with no newline, so it took escape_payload's early return and never
    reached the escape table at all, and the newline test had no backslash.
    Deleting the backslash entry from _ESCAPES passed all 89 tests.

    It is not cosmetic. Escaping the newline without first doubling the
    backslash makes the literal two characters `\` `n` indistinguishable from an
    escaped newline, so unescape_payload turns stored venue bytes into a newline
    that was never sent - silent corruption of a payload the venue was entitled
    to send (a Windows path in an error message is enough).
    """
    payload = 'x\\ny\nz'                    # backslash, 'n', then a REAL newline
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\n" not in escaped, "a real newline survived into the stored line"
    assert unescape_payload(escaped) == payload

    # The realistic shape, and the one a venue actually sends.
    windows_path = '{"a":"C:\\Users"}\ntail'
    escaped, was_escaped = escape_payload(windows_path)
    assert was_escaped is True
    assert "\n" not in escaped
    assert unescape_payload(escaped) == windows_path


def test_a_payload_of_only_backslashes_before_a_newline_roundtrips():
    """Escape order matters most where the backslashes run together."""
    for run in range(1, 6):
        payload = "\\" * run + "\n" + "\\" * run
        escaped, _ = escape_payload(payload)
        assert "\n" not in escaped
        assert unescape_payload(escaped) == payload, f"{run} backslashes"


def test_index_entry_roundtrips():
    entry = IndexEntry(n=7, t_recv_ns=123, t_exch_ms=456,
                       seq={"U": 1, "u": 2}, kind="data", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry


def test_index_entry_allows_missing_exchange_time():
    entry = IndexEntry(n=0, t_recv_ns=1, t_exch_ms=None,
                       seq=None, kind="control", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry
