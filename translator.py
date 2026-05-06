"""Core translation engine that handles chunking, LLM calls, and output assembly."""

import re
import time
import logging
from typing import Generator

from openai import OpenAI

from config import AppConfig

logger = logging.getLogger(__name__)


class Translator:
    """Handles the actual translation logic using a configured LLM backend."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = config.llm
        self.translation = config.translation

        self.client = OpenAI(
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
        )

    def _build_system_prompt(self) -> str:
        """Construct the system prompt from config settings."""
        prompt = (
            f"You are a professional literary translator. "
            f"Translate the following text from {self.translation.source_language} "
            f"to {self.translation.target_language}. "
            f"Preserve the original formatting, paragraph breaks, and style. "
            f"Output only the translated text with no explanations or commentary."
        )
        if self.translation.custom_instructions:
            prompt += f"\n\nAdditional instructions: {self.translation.custom_instructions}"
        return prompt

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks that fit within the token budget."""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = self.translation.chunk_size * 4

        # Try to split on paragraph boundaries first
        paragraphs = re.split(r"(\n{2,})", text)
        chunks: list[str] = []
        current_chunk = ""

        for segment in paragraphs:
            if len(current_chunk) + len(segment) <= max_chars:
                current_chunk += segment
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # If a single paragraph exceeds max_chars, split by sentences
                if len(segment) > max_chars:
                    sentences = re.split(r"(?<=[.!?])\s+", segment)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= max_chars:
                            current_chunk += (" " if current_chunk else "") + sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                else:
                    current_chunk = segment

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _translate_chunk(self, chunk: str, attempt: int = 0) -> str:
        """Send a single chunk to the LLM and return the translation."""
        try:
            response = self.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": chunk},
                ],
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            max_retries = self.config.translation.max_retries
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                return self._translate_chunk(chunk, attempt + 1)
            raise RuntimeError(f"Translation failed after {max_retries} retries: {exc}") from exc

    def translate(self, text: str) -> Generator[str, None, None]:
        """Translate full text, yielding each translated chunk as it completes."""
        chunks = self._split_into_chunks(text)
        total = len(chunks)
        logger.info("Split text into %d chunk(s)", total)

        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Translating chunk %d/%d", idx, total)
            translated = self._translate_chunk(chunk)
            yield translated
            # Small delay between chunks to avoid rate limits
            if idx < total:
                time.sleep(self.translation.request_delay)

    def translate_full(self, text: str) -> str:
        """Convenience method that returns the complete translated text."""
        parts = list(self.translate(text))
        return "\n\n".join(parts)
