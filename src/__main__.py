from .tokenizer import Tokenizer


def main() -> None:
    """Run the function-calling pipeline."""
    text = "Foo © bar 𝌆 bazba ☃ qux"


    tokenizer = Tokenizer.train(text, 10)
    ids = tokenizer.ft_encode(text)
    print(ids)
    retext = tokenizer.ft_decode(ids)
    print(retext)

    print(len(text.encode("utf-8")), "->", len(ids))
    print(retext == text)


if __name__ == "__main__":
    main()
