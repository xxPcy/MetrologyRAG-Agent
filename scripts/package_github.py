from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "MetrologyRAG-Agent"
OUTPUT_DIR = PROJECT_ROOT / "dist"
OUTPUT_ZIP = OUTPUT_DIR / f"{PACKAGE_NAME}-github.zip"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "chroma_db",
    "dist",
    "logs",
    "models",
    "venv",
}
EXCLUDED_EXACT_PATHS = {
    ".env",
    "PRIVATE_GITHUB_UPLOAD.md",
    "data/instruments.csv",
    "作品集.md",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".zip",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    rel_text = rel.as_posix()

    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return False
    if rel_text in EXCLUDED_EXACT_PATHS:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if rel_text.endswith("_export.csv"):
        return False
    if rel.match("data/pdf/*.pdf") or rel.match("data/markdown/*.md"):
        return False
    return True


def iter_package_files() -> list[Path]:
    files = [path for path in PROJECT_ROOT.rglob("*") if path.is_file() and should_include(path)]
    return sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


def build_archive() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        for path in iter_package_files():
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            archive.write(path, f"{PACKAGE_NAME}/{rel}")

    return OUTPUT_ZIP


if __name__ == "__main__":
    archive_path = build_archive()
    size_mb = archive_path.stat().st_size / 1024 / 1024
    print(f"Created {archive_path} ({size_mb:.2f} MB)")
