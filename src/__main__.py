from .tokenizer import Tokenizer
from .parse import load_prompts, load_functions
from .prompt import build_prompt
import json, time
import pydantic
from llm_sdk import Small_LLM_Model
import numpy
from pathlib import Path

def generate(m: Small_LLM_Model, ids: any) -> list:
    STOP = {151643, 151645}
    t0 = time.time()
    gen = []
    for step in range(60):
        logits = m.get_logits_from_input_ids(ids)
        nxt = numpy.argmax(logits)
        if nxt in STOP:
            print("\n[Stopped on", nxt, "after", step, "tokens]")
            break
        ids.append(nxt)
        gen.append(nxt)
    el = time.time() - t0
    return gen


def main() -> None:
    """Run the function-calling pipeline."""

    m = Small_LLM_Model()

    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
        print(f"{s:16} -> {m.encode(s).tolist()[0]}")
        print()

    defs = load_functions("data/input/functions_definition.json")
    prompt = "What is the sum of 2 and 3?"
    full = build_prompt(defs, prompt)

    print("=== prompt tail ===")
    print(full[-200:])
    ids = m.encode(full).tolist()[0]
    print("\nprompt length:", len(ids), "tokens")

    gen = generate(m, ids)
   
    print("\n=== RAW GENERATED TEXT ===")
    data = m.decode(gen)
    print(repr(data))

    out = Path("data/output/functiopn_calling_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    #print("\n%d tokens in %.1fs (%.2fs/token)" % (len(gen), el, el / max(len(gen), 1)))
    try:
        print("json.loads ->", json.loads(m.decode(gen)))
    except Exception as e:
        print("json.loads FAILED ->", type(e).__name__ + ":", e)


if __name__ == "__main__":
    main()
