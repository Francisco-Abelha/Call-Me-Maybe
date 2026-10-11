# tests/test_constraints.py
import json

from src.constraints import allowed_name_tokens, allowed_number_tokens, allowed_string_tokens
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

NUM_VOCAB = Vocab(
    id_to_text={
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "6",
        6: "7",
        7: "8",
        8: "9",
        9: "0",
        10: "-",
        11: ".",
        12: "}",
        13: ",",
        14: "abc",      # decoy: not a number at all
        15: "1x",       # decoy: digit followed by a letter
        16: "²",        # decoy: isdigit() is True but not ASCII
        17: "１",        # decoy: full-width digit
        18: " 1",       # decoy: leading whitespace
        19: '",',       # decoy: structural token from a string context
        20: "",         # empty token
    },
    partial_ids=set(),
)

DIGIT_TEXTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
DECOYS = {"abc", "1x", "²", "１", " 1", '",'}

# Exact allowed set for every reachable state of the number FSM.
NUMBER_STATES = {
    "": DIGIT_TEXTS | {"-"},
    "-": DIGIT_TEXTS,
    "0": {".", ","},
    "-0": {".", ","},
    "1": DIGIT_TEXTS | {".", ","},
    "15": DIGIT_TEXTS | {".", ","},
    "1.": DIGIT_TEXTS,
    "1.5": DIGIT_TEXTS | {","},
    "-3": DIGIT_TEXTS | {".", ","},
    "-3.5": DIGIT_TEXTS | {","},
}


def _num(so_far: str, delimiter: str = ",") -> set[str]:
    """Allowed token texts for a number in the given state."""
    ids = allowed_number_tokens(so_far, delimiter, NUM_VOCAB)
    return {NUM_VOCAB.id_to_text[i] for i in ids}


# --- strings: setup -------------------------------------------------------

STR_VOCAB = Vocab(
    id_to_text={
        0: "a",
        1: "hello",
        2: '"',           # closing quote (terminator)
        3: "\\",          # lone backslash (starts escape)
        4: "n",           # valid after backslash
        5: "\\n",         # backslash + n in one token
        6: 'world"',      # text ending with closing quote
        7: "",            # empty token
        8: "\t",          # control char (0x09)
        9: "\n",          # control char (0x0a)
        10: "x",          # not a valid escape char on its own
        11: "\\t",        # backslash + t in one token
        12: 'ab"cd',      # quote mid-token (invalid)
        13: "\\\\",       # two backslashes (escape + backslash)
        14: '\\"',        # escaped quote (not a terminator)
        15: "ab\\ncd",    # text with escape sequence mid-token
        16: "ab\\",       # text ending with backslash
        17: " ",          # space (valid, ord 0x20)
        18: "\\x",        # invalid escape sequence
    },
    partial_ids=set(),
)


def _str(so_far: str) -> set[str]:
    """Allowed token texts for a string in the given state."""
    ids = allowed_string_tokens(so_far, STR_VOCAB)
    return {STR_VOCAB.id_to_text[i] for i in ids}


# --- function names -------------------------------------------------------

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


# --- numbers: state coverage ---------------------------------------------

def test_number_states() -> None:
    for so_far, expected in NUMBER_STATES.items():
        assert _num(so_far) == expected, f"so_far={so_far!r}"


# --- numbers: the governing invariant ------------------------------------

def test_delimiter_allowed_iff_so_far_is_valid_json() -> None:
    """The number may end exactly when what we have parses as JSON."""
    states = ["", "-", "0", "-0", "1", "15", "1.", "1.5", "-3", "-3.5", "0.5", ".", "-."]
    for so_far in states:
        try:
            json.loads(so_far)
            valid = True
        except ValueError:
            valid = False
        assert ("," in _num(so_far)) == valid, f"so_far={so_far!r} valid={valid}"


# --- numbers: nothing illegal gets through -------------------------------

def test_decoys_never_allowed() -> None:
    for so_far in NUMBER_STATES:
        assert _num(so_far) & DECOYS == set(), f"so_far={so_far!r}"


def test_only_the_given_delimiter_is_allowed() -> None:
    assert "}" not in _num("1", ",")
    assert "," not in _num("1", "}")
    assert "}" in _num("1", "}")
    assert "," in _num("1", ",")


def test_invalid_json_is_unreachable() -> None:
    assert "." not in _num("")       # .5 -- no integer part
    assert "." not in _num("-")      # -.5
    assert "-" not in _num("-")      # --1
    assert "-" not in _num("1")      # 1-2
    assert "." not in _num("1.2")    # 1.2.3 -- second dot
    assert "," not in _num("1.")     # 1. -- trailing dot
    assert "," not in _num("-")      # - alone
    assert "," not in _num("")       # empty value


def test_leading_zeros_blocked() -> None:
    assert _num("0") & DIGIT_TEXTS == set()
    assert _num("-0") & DIGIT_TEXTS == set()


# --- numbers: everything legal stays reachable ---------------------------

def test_valid_numbers_are_reachable() -> None:
    for target in ["0", "5", "15", "100", "-9", "0.5", "-9.5", "1.25"]:
        so_far = ""
        for ch in target:
            assert ch in _num(so_far), f"{target!r} blocked at {so_far!r}"
            so_far += ch
        assert "," in _num(so_far), f"{target!r} cannot terminate"


def test_leading_zeros_blocked_for_runtime_built_strings() -> None:
    """The driver builds so_far with +=, which does not produce interned strings.

    A guard written with `is not` passes when so_far is a source literal and
    fails here, so this case must be built the same way the driver builds it.
    """
    for target in ["0", "-0"]:
        so_far = ""
        for ch in target:
            so_far += ch
        assert _num(so_far) & DIGIT_TEXTS == set(), f"{target!r} allows a leading zero"


# --- strings: basic normal-mode tokens -----------------------------------

def test_string_start_allows_text_and_terminator() -> None:
    allowed = _str("")
    assert "a" in allowed
    assert "hello" in allowed
    assert '"' in allowed        # empty string is valid
    assert " " in allowed


def test_string_mid_allows_continuation() -> None:
    allowed = _str("hel")
    assert "a" in allowed
    assert "hello" in allowed
    assert '"' in allowed


# --- strings: backslash and escape sequences ------------------------------

def test_lone_backslash_allowed_in_normal_mode() -> None:
    assert "\\" in _str("")
    assert "\\" in _str("abc")


def test_after_backslash_only_valid_escape_chars() -> None:
    allowed = _str("\\")
    assert "n" in allowed         # \n
    assert "x" not in allowed     # \x is not valid JSON escape
    assert "a" not in allowed     # \a is not valid JSON escape
    assert "hello" not in allowed # starts with 'h', not a valid escape


def test_backslash_n_single_token() -> None:
    assert "\\n" in _str("")      # valid escape as one token
    assert "\\t" in _str("")


def test_invalid_escape_single_token() -> None:
    assert "\\x" not in _str("")  # \x not valid JSON escape


# --- strings: escaped quote is not a terminator ---------------------------

def test_escaped_quote_allowed() -> None:
    assert '\\"' in _str("")      # produces \" in the string, not termination


def test_escaped_quote_after_backslash() -> None:
    # so_far ends with \, token is " → valid escape, and it's the last char
    # but this is consuming the escape, so it should be allowed
    allowed = _str("\\")
    assert '"' in allowed


# --- strings: double backslash -------------------------------------------

def test_double_backslash_returns_to_normal() -> None:
    assert "\\\\" in _str("")     # \\ is escape + backslash → back to normal
    # after \\, we're in normal mode, so regular chars are allowed
    allowed = _str("\\\\")
    assert "a" in allowed
    assert '"' in allowed


# --- strings: terminator detection ---------------------------------------

def test_quote_at_end_of_token_terminates() -> None:
    assert 'world"' in _str("")
    assert 'world"' in _str("hello ")


def test_quote_mid_token_rejected() -> None:
    assert 'ab"cd' not in _str("")
    assert 'ab"cd' not in _str("hello")


# --- strings: control characters blocked ----------------------------------

def test_control_chars_blocked() -> None:
    assert "\t" not in _str("")
    assert "\n" not in _str("")
    assert "\t" not in _str("abc")


# --- strings: empty token ignored ----------------------------------------

def test_empty_token_ignored() -> None:
    assert "" not in _str("")


# --- strings: compound tokens --------------------------------------------

def test_text_with_escape_mid_token() -> None:
    assert "ab\\ncd" in _str("")  # a, b, \n, c, d — all valid


def test_text_ending_with_backslash() -> None:
    # "ab\" leaves us in escape state — that's valid (next token
    # must start with a valid escape char)
    assert "ab\\" in _str("")


# --- strings: reachability ------------------------------------------------

def test_simple_strings_reachable() -> None:
    so_far = ""
    assert "hello" in _str(so_far)
    so_far += "hello"
    assert '"' in _str(so_far)


def test_string_with_escape_reachable() -> None:
    so_far = ""
    assert "hello" in _str(so_far)
    so_far += "hello"
    assert "\\n" in _str(so_far)
    so_far += "\\n"
    assert '"' in _str(so_far)
