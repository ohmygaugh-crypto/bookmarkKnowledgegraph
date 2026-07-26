import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from api.api import LLMConfig, async_chat, get_llm_config


class AsyncChunks:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class LLMConfigTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_HTTP_REFERER": "http://127.0.0.1:8000",
            "OPENROUTER_APP_NAME": "ThisGuyReadMyMind",
        },
        clear=True,
    )
    def test_openrouter_defaults_to_deepseek_v4_flash(self):
        config = get_llm_config()

        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(
            config.default_headers,
            {
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-OpenRouter-Title": "ThisGuyReadMyMind",
            },
        )

    @patch.dict(os.environ, {"LLM_PROVIDER": "lmstudio"}, clear=True)
    def test_lmstudio_defaults_to_installed_gemma(self):
        config = get_llm_config()

        self.assertEqual(config.provider, "lmstudio")
        self.assertEqual(
            config.base_url, "http://host.docker.internal:1234/v1"
        )
        self.assertEqual(config.model, "google/gemma-4-e4b")
        self.assertEqual(config.api_key, "lm-studio")
        self.assertEqual(config.max_tokens, 1024)

    @patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True)
    def test_openrouter_requires_api_key(self):
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
            get_llm_config()

    @patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}, clear=True)
    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported LLM_PROVIDER"):
            get_llm_config()


class ChatStreamingTests(unittest.IsolatedAsyncioTestCase):
    @patch("api.api.openai.AsyncOpenAI")
    async def test_streams_content_from_typed_openai_chunks(self, async_openai):
        chunks = AsyncChunks(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="OK"))]
                ),
            ]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=chunks)
        async_openai.return_value = client
        config = LLMConfig(
            provider="lmstudio",
            base_url="http://localhost:1234/v1",
            model="google/gemma-4-e4b",
            api_key="lm-studio",
            max_tokens=1024,
            default_headers={},
        )

        streamed = [
            text async for text in async_chat("test query", "test content", config)
        ]

        self.assertEqual(streamed, ["OK"])


if __name__ == "__main__":
    unittest.main()
