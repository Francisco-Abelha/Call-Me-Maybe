from .parse import load_prompts, load_functions
from .prompt import build_prompt
from .constraints import allowed_name_tokens, allowed_number_tokens
from .models import FunctionCall, JSONResponse, FunctionDef
from .vocab import Vocab
import json
# import time
from llm_sdk import Small_LLM_Model
import numpy
from pathlib import Path
import re


max_tokens = 200


def generate(m: Small_LLM_Model, ids: list[int]) -> list[int]:
    STOP = {151643, 151645}
    # t0 = time.time()
    gen = []
    for step in range(max_tokens):
        logits = m.get_logits_from_input_ids(ids)
        nxt = int(numpy.argmax(logits))
        if nxt in STOP:
            print("\n[Stopped on", nxt, "after", step, "tokens]")
            break
        ids.append(nxt)
        gen.append(nxt)
    # el = time.time() - t0
    return gen


def parse_json_response(text: str) -> JSONResponse:
    """Just to check the model end to end, uses the
    'hope the model gets
    it right' aproach. Will be swapped out when i
    build the constrained decoding aproach
    """
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return JSONResponse.model_validate_json(text)


def force(m: Small_LLM_Model, ids: list[int], text: str) -> None:
    """Append the tokens of fixed text to the context, without asking the model."""
    ids.extend(m.encode(text).tolist()[0])


def generate_name(m: Small_LLM_Model, ids: list[int], names: list[str], v: Vocab) -> str:
    """generate a function name, constrained to the given names"""

    so_far = ""
    for _ in range(max_tokens):
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


def generate_number(m: Small_LLM_Model, ids: list[int], delimiter: str, v: Vocab) -> str:
    so_far = ""
    for _ in range(32):
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


def main() -> None:
    """Run the function-calling pipeline."""

    m = Small_LLM_Model()
    v = Vocab.from_file(m.get_path_to_vocab_file())

    defs = load_functions("data/input/functions_definition.json")
    prompts = load_prompts("data/input/function_calling_tests.json")

    by_name: dict[str, FunctionDef] = {}
    for function in defs:
        by_name[function.name] = function

    results: list[FunctionCall] = []

    for prompt in prompts:
        ids = m.encode(build_prompt(defs, prompt)).tolist()[0]
        start = len(ids)

        force(m, ids, '{"name": "')
        name = generate_name(m, ids, list(by_name), v)
        deff = by_name[name]
        allnum = True
        for spec in deff.parameters.values():
            if spec.type != "number":
                allnum = False
        if not deff.parameters:
            force(m, ids, ', "parameters": {}}')
        elif allnum:
            force(m, ids, ', "parameters": {')
            val: dict[str, str] = {}
            param_count = len(deff.parameters)
            for i, (pname, spec) in enumerate(deff.parameters.items()):
                force(m, ids, f'"{pname}": ')
                if i < param_count - 1:
                    delimiter = ","
                else:
                    delimiter = "}"
                val[pname] = generate_number(m, ids, delimiter, v)
            force(m, ids, "}")
        else:
            generate(m, ids)

        text = m.decode(ids[start:])
        print(f"{prompt!r} -> {name}")
        print("  raw:", text)
        try:
            raw = parse_json_response(text)
            results.append(FunctionCall(prompt=prompt, name=raw.name, parameters=raw.parameters))
        except ValueError as e:
            print("  parse failed:", e)

    out = Path("data/output/function_calling_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=4, ensure_ascii=False)
    # print("\n%d tokens in %.1fs (%.2fs/token)" % (len(gen), el, el / max(len(gen), 1)))
    """ try:
        print("json.loads ->", json.loads(m.decode(gen)))
    except Exception as e:
        print("json.loads FAILED ->", type(e).__name__ + ":", e) """


if __name__ == "__main__":
    main()
