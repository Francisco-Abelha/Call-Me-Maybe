from pydantic import BaseModel
import json


def _bytes_to_unicode() -> dict[int, str]:
    """Build the gpt-2 table"""
    visible: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {}
    for b in visible:
        table[b] = chr(b)
    n = 0
    for b in range(256):
        if b not in visible:
            table[b] = chr(256 + n)
            n += 1
    return table


# built once when the file is imported
_BYTE_DECODER: dict[str, int] = {ch: b for b, ch in _bytes_to_unicode().items()}


def byte_level_to_text(token: str) -> str | None:
    bytes_list: list[int] = []
    for char in token:
        bytes_list.append(_BYTE_DECODER[char])
    b = bytes(bytes_list)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


class Vocab(BaseModel):

    id_to_text: dict[int, str]
    partial_ids: set[int]

    @classmethod
    def from_file(cls, path: str) -> "Vocab":

        partial_ids: set[int] = set()

        with open(path, encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
        id_to_text: dict[int, str] = {}
        for token, tok_id in raw.items():
            text = byte_level_to_text(token)
            if text is None:
                partial_ids.add(tok_id)
            else:
                id_to_text[tok_id] = text
        return cls(id_to_text=id_to_text, partial_ids=partial_ids)
