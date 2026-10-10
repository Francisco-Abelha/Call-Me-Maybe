from .vocab import Vocab


DIGITS = "0123456789"


def allowed_name_tokens(so_far: str, names: list[str], vocab: Vocab) -> list[int]:
    allowed: list[int] = []
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        if any(
            name.startswith(so_far + text) for name in names
        ) or (text.endswith('"') and so_far + text[:-1] in names):
            allowed.append(tok_id)
    return allowed


def allowed_number_tokens(so_far: str, delimiter: str, vocab: Vocab) -> list[int]:
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


def allowed_string_tokens(so_far: str, vocab: Vocab) -> list[int]:
    allowed: list[int] = []
    return allowed
