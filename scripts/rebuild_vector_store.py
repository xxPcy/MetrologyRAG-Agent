from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.ingestion.vector_store import build_vector_store, get_chroma_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the MetrologyRAG Chroma vector store.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append/update without deleting the existing Chroma directory first.",
    )
    parser.add_argument(
        "--skip-pdf-convert",
        action="store_true",
        help="Use existing Markdown files and skip PDF-to-Markdown conversion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app_settings = get_settings()

    print(f"Embedding provider: {app_settings.embedding_provider}")
    print(f"Local embedding model: {app_settings.local_embedding_model or '<empty>'}")
    print(f"API embedding model: {app_settings.embedding_model or '<empty>'}")
    print(f"Chroma directory: {app_settings.chroma_dir}")
    print("Rebuilding vector store...")

    _, chunks = build_vector_store(
        rebuild=not args.keep_existing,
        app_settings=app_settings,
        convert_pdfs=not args.skip_pdf_convert,
    )
    chroma_count = get_chroma_count(app_settings)

    print(f"Markdown chunks: {chunks}")
    print(f"Chroma collection count: {chroma_count}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
