import numpy
from llm_sdk import Small_LLM_Model
from .vocab import Vocab
from .constraints import (
    allowed_name_tokens,
    allowed_number_tokens,
    allowed_string_tokens,
    simulate_string,
    allowed_boolean_tokens,
    allowed_integer_tokens
)


MAX_NUMBER_TOKENS = 32
MAX_STRING_TOKENS = 512
BOOLEAN_VALUES = ["true", "false"]


def generate_name(m: Small_LLM_Model, ids: list[int], names: list[str], v: Vocab) -> str:
    """generate a function name, constrained to the given names"""

    so_far = ""
    for _ in range(max((len(n) for n in names), default=0)):
        allowed = allowed_name_tokens(so_far, names, v)
        if not allowed:
            raise ValueError(
                f"so_far: {so_far}"
                f", and names: {names}"
            )
        logits = numpy.array(m.get_logits_from_input_ids(ids))
        nxt = allowed[int(numpy.argmax(logits[allowed]))]
        ids.append(nxt)
        text = v.id_to_text[nxt]
        if text.endswith('"'):
            return so_far + text[:-1]
        so_far += text
    raise RuntimeError("name generation did not finish")


def generate_boolean(m: Small_LLM_Model, ids: list[int], v: Vocab) -> str:
    """Generate a boolean type parameter, constrained to true or false"""

    so_far = ""
    for _ in range(max(len(val) for val in BOOLEAN_VALUES)):
        allowed = allowed_boolean_tokens(so_far, BOOLEAN_VALUES, v)
        if not allowed:
            raise ValueError(f"so_far: {so_far}")
        logits = numpy.array(m.get_logits_from_input_ids(ids))
        nxt = allowed[int(numpy.argmax(logits[allowed]))]
        ids.append(nxt)
        text = v.id_to_text[nxt]
        if (so_far + text) in BOOLEAN_VALUES:
            return so_far + text
        so_far += text
    raise RuntimeError("boolean generation did not finish")


def generate_string(m: Small_LLM_Model, ids: list[int], v: Vocab) -> str:
    """Generate string parameters, constrained to any valid string token
    and dealing with edge cases
    """

    so_far = ""
    for _ in range(MAX_STRING_TOKENS):
        allowed = allowed_string_tokens(so_far, v)
        if not allowed:
            raise ValueError(f"so_far: {so_far}")
        logits = numpy.array(m.get_logits_from_input_ids(ids))
        for tid in allowed:
            if simulate_string(so_far, v.id_to_text[tid]) == "terminated":
                logits[tid] += len(so_far) ** 1.5 * 0.05
        nxt = allowed[int(numpy.argmax(logits[allowed]))]
        ids.append(nxt)
        text = v.id_to_text[nxt]
        if simulate_string(so_far, text) == "terminated":
            return so_far + text[:-1]
        so_far += text
    raise RuntimeError("string generation did not finish")


def generate_number(m: Small_LLM_Model, ids: list[int], delimiter: str, v: Vocab) -> str:
    """Generate a number parameter, constrained to positive or negative numbers
    with no leading zeros
    """

    so_far = ""
    for _ in range(MAX_NUMBER_TOKENS):
        allowed = allowed_number_tokens(so_far, delimiter, v)
        if not allowed:
            raise ValueError(
                f"so_far: {so_far}"
                f", and delimiter: {delimiter}"
            )
        logits = numpy.array(m.get_logits_from_input_ids(ids))
        nxt = allowed[int(numpy.argmax(logits[allowed]))]
        ids.append(nxt)
        text = v.id_to_text[nxt]
        if text == delimiter:
            return so_far
        so_far += text
    raise RuntimeError("number parameter generation did not finish")


def generate_integer(m: Small_LLM_Model, ids: list[int], delimiter: str, v: Vocab) -> str:
    """Generate an integer parameter, constrained to positive or negative integers"""
    so_far = ""
    for _ in range(MAX_NUMBER_TOKENS):
        allowed = allowed_integer_tokens(so_far, delimiter, v)
        if not allowed:
            raise ValueError(
                f"so_far: {so_far}"
                f", and delimiter: {delimiter}"
            )
        logits = numpy.array(m.get_logits_from_input_ids(ids))
        nxt = allowed[int(numpy.argmax(logits[allowed]))]
        ids.append(nxt)
        text = v.id_to_text[nxt]
        if text == delimiter:
            return so_far
        so_far += text
    raise RuntimeError("number parameter generation did not finish")
