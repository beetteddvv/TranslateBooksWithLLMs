"""Configuration management for TranslateBooksWithLLMs.

Loads and validates settings from environment variables / .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str  # openai | anthropic | google | ollama | openrouter
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    request_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class TranslationConfig:
    """Translation job configuration."""

    source_language: str = "auto"
    target_language: str = "French"  # changed to French since that's what I mostly use
    chunk_size: int = 1500  # characters per chunk sent to LLM
    overlap: int = 0        # overlap between chunks (chars)
    preserve_formatting: bool = True
    glossary_path: Optional[str] = None
    system_prompt_override: Optional[str] = None


@dataclass
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig
    translation: TranslationConfig
    output_dir: Path = field(default_factory=lambda: Path("output"))
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))


def _require(key: str) -> str:
    """Return env var value or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return value


def _get_llm_config() -> LLMConfig:
    """Build LLMConfig from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Resolve default model per provider
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-1.5-flash",
        "ollama": "llama3",
        "openrouter": "openai/gpt-4o-mini",
    }
    model = os.getenv("LLM_MODEL", default_models.get(provider, "gpt-4o-mini"))

    # API key is optional for local providers like Ollama
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        request_timeout=int(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("LLM_RETRY_DELAY", "2.0")),
    )


def _get_translation_config() -> TranslationConfig:
    """Build TranslationConfig from environment variables."""
    return TranslationConfig(
        source_la