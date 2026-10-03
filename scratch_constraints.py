from src.constraints import allowed_name_tokens
from src.parse import load_functions
from src.vocab import Vocab

v = Vocab.from_file("/home/fgoncal2/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json")
defs = load_functions("data/input/functions_definition.json")

names: list[str] = []
for function in defs:
    names.append(function.name)

for so_far in ["", "fn", "fn_", "fn_greet", "fn_get_square_root", "fn_add", "fn_xyz"]:
    ids = allowed_name_tokens(so_far, names, v)
    print(repr(so_far), len(ids), [v.id_to_text[i] for i in ids][:20])
