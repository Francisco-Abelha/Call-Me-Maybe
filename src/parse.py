from pydantic import TypeAdapter, ValidationError
from .models import PromptItem, FunctionDef
import sys


def load_prompts(path: str) -> list[str]:
    """Load prompt file"""

    try:
        with open(path) as file:
            items = TypeAdapter(list[PromptItem]).validate_json(file.read())
    except FileNotFoundError as e:
        print("File for the promp doesnt exist: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Valid prompt file but wrong structure: ", e)
        sys.exit(1)
    data = [item.prompt for item in items]
    return data


def load_functions(path: str) -> list[FunctionDef]:
    """Load function definitions file"""

    try:
        with open(path) as file:
            return TypeAdapter(list[FunctionDef]).validate_json(file.read())
    except FileNotFoundError as e:
        print("File for the functions doesnt exist: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Valid functions file but wrong structure: ", e)
        sys.exit(1)
