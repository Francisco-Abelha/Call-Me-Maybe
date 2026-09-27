import json
from pydantic import BaseModel, TypeAdapter


class PromptItem(BaseModel):
    prompt: str


class Parser():

    @classmethod
    def parse(cls, path: str) -> list[str]:
        with open(path) as file:
            items = TypeAdapter(list[PromptItem]).validate_json(file.read())
        data = [item.prompt for item in items]
        return data
    
