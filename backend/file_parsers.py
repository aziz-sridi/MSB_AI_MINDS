"""
AI MINDS — File parsers for various document and media formats.
"""

import pathlib
from typing import Optional

# Supported extensions
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json",
    ".pdf", ".docx",
}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
}
ALL_SUPPORTED = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


def parse_txt(file_path: pathlib.Path) -> str:
    """Parse plain text files (.txt, .md, .log, .csv, .json)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[parser] Failed to parse {file_path.name}: {e}")
        return ""


def parse_pdf(file_path: pathlib.Path) -> str:
    """Parse PDF files using pdfminer."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(file_path))
    except Exception as e:
        print(f"[parser] Failed to parse PDF {file_path.name}: {e}")
        return ""


def parse_docx(file_path: pathlib.Path) -> str:
    """Parse DOCX files using python-docx."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"[parser] Failed to parse DOCX {file_path.name}: {e}")
        return ""


def parse_image(file_path: pathlib.Path) -> str:
    """Return a marker string for image files."""
    return "[IMAGE]"


def get_parser(extension: str):
    """Return the appropriate parser function for a file extension."""
    parsers = {
        ".txt": parse_txt,
        ".md": parse_txt,
        ".log": parse_txt,
        ".csv": parse_txt,
        ".json": parse_txt,
        ".pdf": parse_pdf,
        ".docx": parse_docx,
    }
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        parsers[ext] = parse_image
    return parsers.get(extension.lower())
