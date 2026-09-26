class Tokenizer:
    def __init__(self, merges: dict[tuple[int, int], int]):
        self.merges = merges
        self.vocab = vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

    @classmethod
    def train(cls, text, num_merges):
        ids = list(text.encode("utf-8"))
        merges = {}
        for i in range(num_merges):
            stats = cls.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            # if pair count is below 2, then stop
            if stats[top_pair] < 2:
                break
            idx = 256 + i
            ids = cls.merge(ids, top_pair, idx)
            merges[top_pair] = idx
        return cls(merges)

    @staticmethod
    def get_stats(ids: list[int]) -> dict:
        """get a list of encoded tokens along with the number of times they appear"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids, pair, idx) -> list[int]:
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i+= 2
            else:
                newids.append(ids[i])
                i+= 1
        return newids

    def ft_encode(self, text: str) -> list[int]:
        tokens = text.encode("utf-8", errors="replace")
        tokens = list(map(int, tokens))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def ft_decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
        