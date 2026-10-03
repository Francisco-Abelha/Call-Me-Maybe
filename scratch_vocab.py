from src.vocab import Vocab, byte_level_to_text, _bytes_to_unicode

t = _bytes_to_unicode()
assert len(t) == 256 and len(set(t.values())) == 256
assert t[32] == "Ġ" and t[10] == "Ċ"
assert byte_level_to_text("Ġthe") == " the"
assert byte_level_to_text("Ã©") == "é"
assert byte_level_to_text("Ã") is None

v = Vocab.from_file("/home/fgoncal2/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json")
assert len(v.id_to_text) + len(v.partial_ids) == 151643
print(v.id_to_text[279], "|", repr(v.id_to_text[198]), "|", len(v.partial_ids), "partial")
