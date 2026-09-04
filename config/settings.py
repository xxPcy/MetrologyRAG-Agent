from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _default_instruments_path() -> Path:
    private_catalog = PROJECT_ROOT / "data" / "instruments.csv"
    if private_catalog.exists():
        return private_catalog
    return PROJECT_ROOT / "data" / "instruments_demo.csv"


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdf"
    markdown_dir: Path = PROJECT_ROOT / "data" / "markdown"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    instruments_path: Path = PROJECT_ROOT / "data" / "instruments.csv"
    evaluation_cases_path: Path = PROJECT_ROOT / "data" / "evaluation_cases.json"
    demo_standard_path: Path = PROJECT_ROOT / "data" / "demo_standard.md"
    rules_path: Path = PROJECT_ROOT / "config" / "rules.yaml"
    log_dir: Path = PROJECT_ROOT / "logs"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    collection_name: str = "metrology_standards"

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    agent_max_steps: int = 8

    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"
    local_embedding_model: str = ""
    hash_embedding_dimensions: int = 384
    embedding_batch_size: int = 10

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @property
    def embedding_api_configured(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_model)


def get_settings() -> AppSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    return AppSettings(
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
        collection_name=os.getenv("CHROMA_COLLECTION", "metrology_standards"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 8),
        instruments_path=_env_path("INSTRUMENTS_PATH", _default_instruments_path()),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", ""),
        hash_embedding_dimensions=_env_int("HASH_EMBEDDING_DIMENSIONS", 384),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 10),
    )


settings = get_settings()
