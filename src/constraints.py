from .vocab import Vocab


DIGITS = "0123456789"


def allowed_name_tokens(so_far: str, names: list[str], vocab: Vocab) -> list[int]:
    """Oracle for the function name tokens. Returns valid tokens ids"""

    allowed: list[int] = []
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        if any(
            name.startswith(so_far + text) for name in names
        ) or (text.endswith('"') and so_far + text[:-1] in names):
            allowed.append(tok_id)
    return allowed


def allowed_boolean_tokens(so_far: str, values: list[str], vocab: Vocab) -> list[int]:
    """Oracle for the boolean type tokens. Returns valid tokens ids"""
    allowed: list[int] = []
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        if any(
            value.startswith(so_far + text) for value in values
        ) or (so_far + text in values):
            allowed.append(tok_id)
    return allowed


def allowed_number_tokens(so_far: str, delimiter: str, vocab: Vocab) -> list[int]:
    """Oracle for the number tokens. Returns valid tokens ids"""

    allowed: list[int] = []
    has_digit = any(ch in DIGITS for ch in so_far)
    has_dot = "." in so_far
    last = so_far[-1] if so_far else ""
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        elif all(ch in DIGITS for ch in text) and so_far != "0" and so_far != "-0":
            allowed.append(tok_id)
        elif so_far == "" and text == "-":
            allowed.append(tok_id)
        elif text == "." and (has_digit and not has_dot):
            allowed.append(tok_id)
        elif text == delimiter and (has_digit and last not in "-."):
            allowed.append(tok_id)
    return allowed


def simulate_string(so_far: str, token_text: str) -> str | None:
    """Return the state after appending token_text, or None if invalid.
    State is either 'normal', 'escape' or 'terminate'.
    """

    state = "normal"
    for ch in so_far:
        if state == "escape":
            state = "normal"
        elif ch == "\\":
            state = "escape"
    for i, ch in enumerate(token_text):
        if state == "escape":
            if ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
                state = "normal"
            else:
                return None
        else:
            if ch == '\\':
                state = "escape"
            elif ch == '"':
                if i == len(token_text) - 1:
                    return "terminated"
                else:
                    return None
            elif ord(ch) < 0x20:
                return None
    return state


def allowed_string_tokens(so_far: str, vocab: Vocab) -> list[int]:
    """Oracle for the string tokens. Returns valid tokens ids."""

    allowed: list[int] = []
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        if simulate_string(so_far, text):
            allowed.append(tok_id)
    return allowed
