from .models import FunctionDef


def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Build the prompt to be fed into the llm so it knows where
    in the conversation it is supposed to start inputing tokens
    """

    lines = []
    for d in functions:
        args = ", ".join(k + ": " + v.type for k, v in d.parameters.items())
        lines.append("- " + d.name + "(" + args + ") -- " + d.description)
    sig = "\n".join(lines)
    full = (
        '<|im_start|>system\nYou translate requests into function calls. '
        'Reply with only JSON: {"name": ..., "parameters": {...}}\n'
        "Available functions:\n" + sig + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return full
