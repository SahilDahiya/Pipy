from pi_ai.utils.sanitize_unicode import sanitize_surrogates


def test_sanitize_removes_unpaired_surrogates():
    # Unpaired high surrogate + unpaired low surrogate
    text = "hello" + "\ud83d" + "world" + "\udc00"
    sanitized = sanitize_surrogates(text)
    assert "\ud83d" not in sanitized
    assert "\udc00" not in sanitized
    assert sanitized == "helloworld"


def test_sanitize_keeps_valid_pairs():
    text = "emoji \U0001F648"
    sanitized = sanitize_surrogates(text)
    assert sanitized == text
