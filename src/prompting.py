"""Build the text given to the model before generation.

Uses Qwen3's chat template with thinking disabled. The prompt only helps
the model pick the right function and values; the output format is
guaranteed by the constraints, not by the prompt.
"""

from src.models import FunctionDefinition

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NO_THINKING = "<think>\n\n</think>\n\n"
SYSTEM_ROLE = "You translate user requests into function calls."


def _describe(fn: FunctionDefinition) -> str:
    """One line per function, e.g. "- fn_greet(name: string): ..."."""
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in fn.parameters.items())
    return f"- {fn.name}({params}): {fn.description}"


def build_prompt(functions: list[FunctionDefinition], user_prompt: str) -> str:
    """Build the full chat prompt for one request.

    Args:
        functions: available functions, listed in file order.
        user_prompt: the request, copied unchanged.

    Returns:
        The prompt, ending right after the empty thinking block so the
        model starts writing the answer.
    """
    function_lines = "\n".join(_describe(fn) for fn in functions)
    system = f"{SYSTEM_ROLE}\nAvailable functions:\n{function_lines}"
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user_prompt}{IM_END}\n"
        f"{IM_START}assistant\n{NO_THINKING}"
    )
