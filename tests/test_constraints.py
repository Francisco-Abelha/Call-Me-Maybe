# tests/test_constraints.py
from src.constraints import allowed_name_tokens
from src.vocab import Vocab

NAMES = ["fn_add_numbers", "fn_greet", "fn_get_square_root"]

VOCAB = Vocab(
    id_to_text={
        0: "fn",
        1: "_g",
        2: "reet",
        3: '"',
        4: '",',
        5: "rab",
        6: "",          # empty token
        7: "fn_greet",  # whole name in one token
        8: 'reet"',     # finishes the name and closes it
        9: " fn",       # leading space
    },
    partial_ids=set(),
)


def test_start_allows_prefixes() -> None:
    assert sorted(allowed_name_tokens("", NAMES, VOCAB)) == [0, 7]


def test_middle_of_name() -> None:
    assert sorted(allowed_name_tokens("fn", NAMES, VOCAB)) == [1]


def test_completing_name() -> None:
    assert sorted(allowed_name_tokens("fn_g", NAMES, VOCAB)) == [2, 8]


def test_closing_quote_only_after_full_name() -> None:
    assert allowed_name_tokens("fn_greet", NAMES, VOCAB) == [3]
    assert 3 not in allowed_name_tokens("fn_gr", NAMES, VOCAB)


def test_dead_end_returns_nothing() -> None:
    assert allowed_name_tokens("fn_x", NAMES, VOCAB) == []
