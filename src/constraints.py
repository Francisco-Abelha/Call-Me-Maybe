from .vocab import Vocab


def allowed_name_tokens(so_far: str, names: list[str], vocab: Vocab) -> list[int]:
    allowed: list[int] = []
    for (tok_id, text) in vocab.id_to_text.items():
        if not text:
            continue
        if any (name.startswith(so_far + text) for name in names) or (text.endswith('"') and so_far + text[:-1] in names):
            allowed.append(tok_id)
    return allowed


def allowed_number_tokens(so_far: str, vocab: Vocab) -> list[int]:
    pass


def allowed_string_tokens(so_far: str, vocab: Vocab) -> list[int]:
    pass
