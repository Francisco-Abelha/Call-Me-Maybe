from .parse import load_prompts, load_functions
from .prompt import build_prompt
from .constraints import (
    allowed_name_tokens,
    allowed_number_tokens,
    allowed_string_tokens,
    simulate_string,
    allowed_boolean_tokens
)
from .models import FunctionCall, FunctionDef
from .vocab import Vocab
import json
from llm_sdk import Small_LLM_Model
import numpy
from pathlib import Path
import sys
import argparse


MAX_NUMBER_TOKENS = 32
MAX_STRING_TOKENS = 512
BOOLEAN_VALUES = ["true", "false"]


def force(m: Small_LLM_Model, ids: list[int], text: str) -> None:
    """Append the tokens of fixed text to the context, without asking the model."""

    ids.extend(m.encode(text).tolist()[0])


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


def main() -> None:
    """Run the function-calling pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    try:
        m = Small_LLM_Model()
        v = Vocab.from_file(m.get_path_to_vocab_file())
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Failed to load model or vocabulary: {e}")
        sys.exit(1)

    defs = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    by_name: dict[str, FunctionDef] = {}
    for function in defs:
        by_name[function.name] = function

    results: list[FunctionCall] = []

    for prompt in prompts:
        try:
            ids = m.encode(build_prompt(defs, prompt)).tolist()[0]

            force(m, ids, '{"name": "')
            name = generate_name(m, ids, list(by_name), v)
            deff = by_name[name]
            val: dict[str, float | str | bool] = {}
            if not deff.parameters:
                force(m, ids, ', "parameters": {}}')
            else:
                force(m, ids, ', "parameters": {')
                param_count = len(deff.parameters)
                for i, (pname, spec) in enumerate(deff.parameters.items()):
                    force(m, ids, f'"{pname}": ')
                    if i < param_count - 1:
                        delimiter = ","
                    else:
                        delimiter = "}"
                    if spec.type == "number":
                        val[pname] = float(generate_number(m, ids, delimiter, v))
                    elif spec.type == "string":
                        force(m, ids, '"')
                        val[pname] = generate_string(m, ids, v)
                        force(m, ids, delimiter)
                    elif spec.type == "boolean":
                        val[pname] = generate_boolean(m, ids, v) == "true"
                        force(m, ids, delimiter)
                    else:
                        raise ValueError(f"Unknown parameter type: {spec.type}")
                force(m, ids, "}")

            print(f"{prompt!r} -> {name} {val}")
            results.append(
                FunctionCall(prompt=prompt, name=name, parameters=val)
            )
        except (ValueError, RuntimeError) as e:
            print(f"Prompt: {prompt}, {e}")
            continue

    try:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=4, ensure_ascii=False)
    except (PermissionError, OSError) as e:
        print(f"Failed to generate output file: {e}")


if __name__ == "__main__":
    main()
