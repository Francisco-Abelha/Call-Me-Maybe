*This project has been created as part of the 42 curriculum by fgoncal2*

# Call-Me-Maybe

## Description
Call-Me-Maybe is a program that translates natural language into structured JSON
function calls using a small LLM with constrained decoding.

An approach using constrained decoding has a 100% guarantee that the JSON code generated is valid, whereas without it we would be only getting a 30% success rate.

## Instructions

```bash
- uv sync
- make run (default paths)
- uv run python -m src --functions_definition ... --input ... --output ... (custom paths)
- make lint, make clean, etc.
```

## Algorithm
The program first builds a prompt that lets the LLM know where to situate itself in the conversation, leaving what is expected as its response empty so that it starts inserting tokens in that spot.

Certain portions of the response are fixed no matter what the input prompt is so i force the structure on those first and then use the oracle methods to match prefixes and get all the valid token ids for that given position, using different oracles for function names, numbers, strings or boolean types.
Invalid tokens are masked by only allowing the IDs returned by the oracle.

## Design decisions
- Why per-type oracles instead of generic JSON grammar?

Each type has different rules: numbers follow a mathematical grammar, strings need escape handling, booleans are just two values. A generic JSON grammar would be far more complex to implement.

- Why length bias for string termination

For some cases the LLM was always finding valid values along side the terminator token so it would loop an unreasonable amount of time, generating them long after finding the required ones. Length bias introduces an increasing value to the terminator token's weight in the math that the program does on the IDS table to choose a token.

- Why force() for JSON structure instead of generating it

Certain parts of the prompt dont change so it is more cost and time effective to force that structure so the LLM doesnt have to concern itself with generating those extra tokens.

- Why Pydantic for validation

Its required by the subject and it gives me automatic JSON parsing and type validation for input files and data models.

## Performance Analysis
100% valid JSON, near-perfect function selection, 4-5 minutes on CPU. Unconstrained decoding with the same model achieves only 30% guarantee of Valid JSON and takes a longer time to generate.

## Challenges
Some problems i ran into were:
- String termination problem where the vowel loop in the 10th prompt ran infinitely (prompted me to introduce length bias in strings)
- Escape sequence handling: a " inside a token could be either a string terminator or part of an escape sequence like \\". I couldnt just check the last character, i needed to know if the preceding \ is itself escaped. This led to tracking state (normal vs escape) across the entire so_far string, since `\\\"` has the quote escaped but `\\\\"` does not.
- Multi-character tokens required FSM simulation: The LLM's vocabulary contains tokens like hello, \\\n, or world" – not just single characters. I can't check a token with a simple rule like "is this character valid". I need to simulate the entire FSM across every character in the token to determine if the token as a whole is valid. A token might start valid but contain an illegal character mid-way, or end with " making it a terminator.

## Testing strategy
To test my program while building it i implemented unit testing for the oracles with synthetic vocabs.
To learn the whole pipeline in the begining i first build the program without using constrained decoding and then started adding the extra steps that would make up the constrained decoding strategy.
```bash
uv run pytest # Use this command to run the unit tests
```

## Example usage
  ```bash
  uv run python -m src
  'What is the sum of 2 and 3?' -> fn_add_numbers {'a': '2', 'b': '3'}
  'Greet shrek' -> fn_greet {'name': 'shrek'}
  'Reverse the string hello' -> fn_reverse_string {'s': 'hello'}
  ```

## Resources
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) — string and
number grammar rules
- [GPT-2 byte-level
encoding](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer)
— vocabulary decoding approach
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B) — the LLM used
- [Andrej Karpathy — Let's build the GPT
Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — understanding BPE tokenization
- [Andrej Karpathy — Let's build GPT from
scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — understanding LLM generation and
logits

### AI usage
  AI (Claude) was used as a learning companion throughout development:
  - Understanding constrained decoding concepts and FSM design
  - Debugging oracle logic (escape state tracking, multi-character token simulation)
  - Generating unit test scaffolding for the string oracle
  - Reviewing code for correctness after each implementation step