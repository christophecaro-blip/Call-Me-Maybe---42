"""Pydantic models describing the program's input and output data."""

from typing import Literal

from pydantic import BaseModel

ParamType = Literal["number", "string", "integer", "boolean"]


class ParameterSpec(BaseModel):
    """Declared type of a function parameter or return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """One callable function, as listed in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """One generated function call, written to the output file."""

    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]
