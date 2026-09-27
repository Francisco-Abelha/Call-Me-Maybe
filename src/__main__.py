from .tokenizer import Tokenizer
from .parse import Parser
import json


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

    print("---------")
    path = "data/input/function_calling_tests.json"
    try: 
        data = Parser.parse(path)
        for elem in data:
            print(elem)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
