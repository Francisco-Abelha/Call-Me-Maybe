from pydantic import TypeAdapter
from .models import PromptItem, FunctionDef


def load_prompts(path: str) -> list[str]:
    with open(path) as file:
        items = TypeAdapter(list[PromptItem]).validate_json(file.read())
    data = [item.prompt for item in items]
    return data
    

def load_functions(path: str) -> list[FunctionDef]:
    with open(path) as file:
        return TypeAdapter(list[FunctionDef]).validate_json(file.read())
    