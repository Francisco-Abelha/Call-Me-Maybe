from typing import Literal

from pydantic import BaseModel


class PromptItem(BaseModel):
    """single test prompt"""
    prompt: str


class ParamSpec(BaseModel):
    """Response parameter"""
    type: Literal["number", "string", "boolean", "integer"]


class FunctionDef(BaseModel):
    """Function definition structure"""
    name: str
    description: str
    parameters: dict[str, ParamSpec]


class FunctionCall(BaseModel):
    """Function call structure"""
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
