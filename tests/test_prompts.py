from __future__ import annotations

import pytest

from codebook_lab.prompts import (
    PromptContext,
    get_prompt_wrapper,
    list_prompt_wrappers,
    register_prompt_wrapper,
    render_prompt,
)


def _make_context(**overrides) -> PromptContext:
    defaults = dict(
        section_name="Section",
        section_instruction="Annotate carefully.",
        annotation_name="Label",
        tooltip="Some guidance.",
        annotation_type="dropdown",
        options=["a", "b"],
        min_value=None,
        max_value=None,
        label_options=None,
        example="",
        text="Sample text to annotate.",
        use_examples=False,
        response_instructions="Return JSON.",
        core_prompt="Core prompt content.",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


class TestPromptRegistry:
    def test_list_includes_builtins(self):
        wrappers = list_prompt_wrappers()
        assert "standard" in wrappers
        assert "persona" in wrappers
        assert "CoT" in wrappers

    def test_get_builtin(self):
        wrapper = get_prompt_wrapper("standard")
        assert callable(wrapper)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt wrapper"):
            get_prompt_wrapper("nonexistent_wrapper_xyz")

    def test_register_and_use(self):
        def my_wrapper(ctx: PromptContext) -> str:
            return f"custom: {ctx.core_prompt}"

        name = "_test_register_custom"
        register_prompt_wrapper(name, my_wrapper)
        try:
            assert name in list_prompt_wrappers()
            result = render_prompt(name, _make_context())
            assert result.startswith("custom:")
        finally:
            # Clean up to avoid polluting other tests
            from codebook_lab.prompts import _PROMPT_WRAPPERS
            _PROMPT_WRAPPERS.pop(name, None)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_prompt_wrapper("standard", lambda ctx: "")

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_prompt_wrapper("", lambda ctx: "")


class TestBuiltinWrappers:
    def test_standard_includes_text(self):
        ctx = _make_context()
        result = render_prompt("standard", ctx)
        assert ctx.text in result
        assert ctx.core_prompt in result

    def test_persona_includes_expert_framing(self):
        ctx = _make_context()
        result = render_prompt("persona", ctx)
        assert "expert" in result.lower()
        assert ctx.text in result

    def test_cot_includes_step_by_step(self):
        ctx = _make_context()
        result = render_prompt("CoT", ctx)
        assert "step" in result.lower()
        assert ctx.text in result

    def test_callable_wrapper_passthrough(self):
        ctx = _make_context()
        result = render_prompt(lambda c: f"direct:{c.text}", ctx)
        assert result == f"direct:{ctx.text}"
