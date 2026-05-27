"""Embedding generation -- Gemini API + local LM Studio (OpenAI compatible)."""

from __future__ import annotations

from collections.abc import Sequence

from .helpers import logger


class EmbeddingsMixin:
    def _get_embedding_client(self):
        if self._embedding_client is not None:
            return self._embedding_client
        try:
            api_key = str(getattr(self.config, "api_key", "") or "").strip()
            if not api_key or api_key.upper().startswith("YOUR_"):
                return None
            from google import genai
            self._embedding_client = genai.Client(api_key=api_key)
            return self._embedding_client
        except Exception as e:
            logger.debug(f"Discord RAG Gemini client failed: {e}")
            return None

    def generate_embedding(self, text: str) -> list[float] | None:
        if self.provider == "local":
            return self._embed_local(text)
        return self._embed_gemini(text)

    def generate_embeddings_batch(self, texts: Sequence[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if self.provider == "local":
            return self._embed_local_batch(list(texts))
        return self._embed_gemini_batch(list(texts))

    def _embed_gemini(self, text: str) -> list[float] | None:
        client = self._get_embedding_client()
        if client is None:
            return None
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=gtypes.EmbedContentConfig(output_dimensionality=self.embedding_dims),
            )
            return result.embeddings[0].values if result.embeddings else None
        except Exception as e:
            logger.debug(f"Discord RAG Gemini embedding failed: {e}")
            return None

    def _embed_gemini_batch(self, texts: list[str]) -> list[list[float] | None]:
        client = self._get_embedding_client()
        if client is None:
            return [None] * len(texts)
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=gtypes.EmbedContentConfig(output_dimensionality=self.embedding_dims),
            )
            return [emb.values if emb else None for emb in result.embeddings]
        except Exception as e:
            logger.debug(f"Discord RAG Gemini batch embedding failed: {e}")
            return [None] * len(texts)

    def _embed_local(self, text: str) -> list[float] | None:
        if self._httpx_client is None:
            return None
        try:
            resp = self._httpx_client.post(
                f"{self.lm_studio_url}/v1/embeddings",
                json={"model": self.local_embedding_model, "input": [text]},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.debug(f"Discord RAG local embedding failed: {e}")
            return None

    def _embed_local_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float] | None]:
        if self._httpx_client is None:
            return [None] * len(texts)
        embeddings: list[list[float] | None] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self._httpx_client.post(
                    f"{self.lm_studio_url}/v1/embeddings",
                    json={"model": self.local_embedding_model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                ordered = sorted(data["data"], key=lambda item: item["index"])
                embeddings.extend([item["embedding"] for item in ordered])
            return embeddings
        except Exception as e:
            logger.debug(f"Discord RAG local batch embedding failed: {e}")
            return embeddings + [None] * (len(texts) - len(embeddings))
