from .parse import load_prompts, load_functions
from .prompt import build_prompt
from .models import FunctionCall, FunctionDef
from .vocab import Vocab
import json
from llm_sdk import Small_LLM_Model
from pathlib import Path
import sys
import argparse
from .generators import (
    generate_boolean,
    generate_name,
    generate_number,
    generate_string,
    generate_integer
)


def force(m: Small_LLM_Model, ids: list[int], text: str) -> None:
    """Append the tokens of fixed text to the context, without asking the model."""

    ids.extend(m.encode(text).tolist()[0])


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
            val: dict[str, int | float | str | bool] = {}
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
                    elif spec.type == "integer":
                        val[pname] = int(generate_integer(m, ids, delimiter, v))
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
