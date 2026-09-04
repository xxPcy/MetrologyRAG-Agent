from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from langchain_core.embeddings import Embeddings

from config.settings import AppSettings, settings
from src.utils.logger import get_logger


logger = get_logger(__name__)
_EMBEDDING_CACHE: dict[tuple[object, ...], Embeddings] = {}


class HashEmbeddings(Embeddings):
    """Small deterministic embedding fallback for API-less demos and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> Iterable[str]:
        normalized = text.lower()
        base_tokens = re.findall(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]", normalized)
        for token in base_tokens:
            yield token
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
        for index in range(len(chinese_chars) - 1):
            yield "".join(chinese_chars[index : index + 2])


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def get_embedding_model(app_settings: AppSettings = settings) -> Embeddings:
    provider = app_settings.embedding_provider.lower()
    cache_key = (
        provider,
        app_settings.local_embedding_model,
        app_settings.embedding_base_url,
        app_settings.embedding_model,
        bool(app_settings.embedding_api_key),
        app_settings.embedding_batch_size,
        app_settings.hash_embedding_dimensions,
    )
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    if provider in {"auto", "sentence_transformers"} and app_settings.local_embedding_model:
        try:
            logger.info("Using local sentence-transformers embedding: %s", app_settings.local_embedding_model)
            embedding = SentenceTransformerEmbeddings(app_settings.local_embedding_model)
            _EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        except Exception as exc:
            logger.warning("Local embedding initialization failed: %s", exc)
            if provider == "sentence_transformers":
                raise

    if provider in {"auto", "openai"} and app_settings.embedding_api_configured:
        try:
            from langchain_openai import OpenAIEmbeddings

            kwargs = {
                "model": app_settings.embedding_model,
                "api_key": app_settings.embedding_api_key,
                "chunk_size": app_settings.embedding_batch_size,
                "check_embedding_ctx_length": False,
            }
            if app_settings.embedding_base_url:
                kwargs["base_url"] = app_settings.embedding_base_url
            embedding = OpenAIEmbeddings(**kwargs)
            _EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        except Exception as exc:
            logger.warning("Embedding API initialization failed: %s", exc)
            if provider == "openai":
                raise

    if provider == "sentence_transformers":
        raise ValueError("LOCAL_EMBEDDING_MODEL is required for sentence_transformers.")

    logger.info("Using deterministic HashEmbeddings fallback.")
    embedding = HashEmbeddings(dimensions=app_settings.hash_embedding_dimensions)
    _EMBEDDING_CACHE[cache_key] = embedding
    return embedding
