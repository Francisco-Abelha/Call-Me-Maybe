from .tokenizer import Tokenizer
from .parse import load_prompts, load_functions
import json, time
import pydantic
from llm_sdk import Small_LLM_Model
import numpy


def main() -> None:
    """Run the function-calling pipeline."""
    """ text = "Foo © bar 𝌆 bazba ☃ qux"


    tokenizer = Tokenizer.train(text, 10)
    ids = tokenizer.ft_encode(text)
    print(ids)
    retext = tokenizer.ft_decode(ids)
    print(retext)

    print(len(text.encode("utf-8")), "->", len(ids))
    print(retext == text) """

    m = Small_LLM_Model()

    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
        print(f"{s:16} -> {m.encode(s).tolist()[0]}")
        print()

    defs = json.load(open("data/input/functions_definition.json"))
    lines = []
    for d in defs:
        args = ", ".join(k + ": " + v["type"] for k, v in d["parameters"].items())
        lines.append("- " + d["name"] + "(" + args + ") -- " + d["description"])
    sig = "\n".join(lines)

    prompt = "What is the sum of 2 and 3?"
    full = (
        '<|im_start|>system\nYou translate requests into function calls. '
        'Reply with only JSON: {"name": ..., "parameters": {...}}\n'
        "Available functions:\n" + sig + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    print("=== prompt tail ===")
    print(full[-200:])
    ids = m.encode(full).tolist()[0]
    print("\nprompt length:", len(ids), "tokens")

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

    print("\n=== RAW GENERATED TEXT ===")
    print(repr(m.decode(gen)))
    print("\n%d tokens in %.1fs (%.2fs/token)" % (len(gen), el, el / max(len(gen), 1)))
    try:
        print("json.loads ->", json.loads(m.decode(gen)))
    except Exception as e:
        print("json.loads FAILED ->", type(e).__name__ + ":", e)
    print("---------")
    path = "data/input/function_calling_tests.json"
    try: 
        data = load_prompts(path)
        for elem in data:
            print(elem)
    except (FileNotFoundError, pydantic.ValidationError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
