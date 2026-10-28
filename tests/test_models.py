"""Tests for the pydantic data models."""

import pytest
from pydantic import ValidationError

from src.models import FunctionCall, FunctionDefinition, ParameterSpec


def test_parameter_spec_accepts_known_types() -> None:
    """The four supported JSON types are accepted."""
    for name in ("number", "integer", "string", "boolean"):
        assert ParameterSpec(type=name).type == name


def test_parameter_spec_rejects_unknown_type() -> None:
    """An unknown type fails at load time (fail fast)."""
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate({"type": "banana"})


def test_function_definition_keeps_parameter_order() -> None:
    """Parameters keep the file order: it drives the generated keys."""
    fn = FunctionDefinition.model_validate({
        "name": "fn",
        "description": "d",
        "parameters": {"b": {"type": "number"}, "a": {"type": "string"}},
        "returns": {"type": "number"},
    })
    assert list(fn.parameters) == ["b", "a"]


def test_function_call_keeps_value_types() -> None:
    """Values keep their exact type: bool stays bool, int stays int."""
    call = FunctionCall(prompt="p", name="fn",
                        parameters={"flag": True, "n": 2, "x": 2.5})
    assert call.parameters == {"flag": True, "n": 2, "x": 2.5}
    assert type(call.parameters["flag"]) is bool
    assert type(call.parameters["n"]) is int
