from __future__ import annotations

DEFAULT_CHAT_MODE = "per_text"
DEFAULT_COUNTRY_ISO_CODE = "USA"
DEFAULT_MODEL = "gemma3:270m"
DEFAULT_PROMPT_TYPE = "standard"
DEFAULT_REASONING = None
DEFAULT_TASK = "policy-sentiment"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = None
DEFAULT_USE_EXAMPLES = True

CHAT_MODES = ("per_text", "per_query", "continuous")


def normalize_chat_mode(value) -> str:
    """Return a supported chat-history policy."""
    normalized = str(value or DEFAULT_CHAT_MODE).strip().lower().replace("-", "_")
    aliases = {
        "text": "per_text",
        "row": "per_text",
        "per_row": "per_text",
        "query": "per_query",
        "fresh": "per_query",
        "stateless": "per_query",
        "one_chat": "continuous",
        "single_chat": "continuous",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CHAT_MODES:
        available = ", ".join(CHAT_MODES)
        raise ValueError(f"Unknown chat_mode '{value}'. Available chat modes: {available}.")
    return normalized


def normalize_reasoning(value) -> bool | str | None:
    """Normalize user/config values for Ollama reasoning mode."""
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip()
    lower = normalized.lower()
    if lower in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if lower in {"false", "0", "no", "n", "f", "off"}:
        return False
    return normalized
