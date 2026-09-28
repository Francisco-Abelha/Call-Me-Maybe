from pydantic import BaseModel, TypeAdapter


class PromptItem(BaseModel):
    prompt: str


class ParamSpec(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamSpec]


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
