from .tokenizer import Tokenizer
from .parse import load_prompts, load_functions
from .prompt import build_prompt
from .models import FunctionCall
import json, time
import pydantic
from llm_sdk import Small_LLM_Model
import numpy
from pathlib import Path
import re

max_tokens = 200

def generate(m: Small_LLM_Model, ids: any) -> list:
    STOP = {151643, 151645}
    t0 = time.time()
    gen = []
    for step in range(max_tokens):
        logits = m.get_logits_from_input_ids(ids)
        nxt = int(numpy.argmax(logits))
        if nxt in STOP:
            print("\n[Stopped on", nxt, "after", step, "tokens]")
            break
        ids.append(nxt)
        gen.append(nxt)
    el = time.time() - t0
    return gen


def parse_json_response(text: str):
    """Just to check the model end to end, uses the 
    'hope the model gets
    it right' aproach. Will be swapped out when i 
    build the constrained decoding aproach
    """
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)



def main() -> None:
    """Run the function-calling pipeline."""

    m = Small_LLM_Model()

    defs = load_functions("data/input/functions_definition.json")
    prompts = load_prompts("data/input/function_calling_tests.json")
    results: list[FunctionCall] = []

    for prompt in prompts:
        ids = m.encode(build_prompt(defs, prompt)).tolist()[0]
        text = m.decode(generate(m, ids))
        results.append(FunctionCall(prompt=prompt, **parse_json_response(text)))

    out = Path("data/output/function_calling_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=4, ensure_ascii=False)
    #print("\n%d tokens in %.1fs (%.2fs/token)" % (len(gen), el, el / max(len(gen), 1)))
    """ try:
        print("json.loads ->", json.loads(m.decode(gen)))
    except Exception as e:
        print("json.loads FAILED ->", type(e).__name__ + ":", e) """


if __name__ == "__main__":
    main()
