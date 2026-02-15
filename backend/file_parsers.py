"""
GigaMind — File parsers for various document and media formats.
"""

import pathlib
from typing import Dict, Optional

# Supported extensions
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json",
    ".pdf", ".docx",
}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
}
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
}
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv",
}
ALL_SUPPORTED = (
    SUPPORTED_TEXT_EXTENSIONS
    | SUPPORTED_IMAGE_EXTENSIONS
    | SUPPORTED_AUDIO_EXTENSIONS
    | SUPPORTED_VIDEO_EXTENSIONS
)

_WHISPER_MODEL_CACHE = {}


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


def _get_whisper_model(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    if key in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[key]
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _WHISPER_MODEL_CACHE[key] = model
    return model


def _transcribe_with_whisper(file_path: pathlib.Path, transcription_cfg: Dict) -> Optional[str]:
    if not transcription_cfg.get("enabled", True):
        return None

    try:
        model_name = transcription_cfg.get("model", "small")
        device = transcription_cfg.get("device", "cpu")
        compute_type = transcription_cfg.get("compute_type", "int8")
        language = (transcription_cfg.get("language") or "").strip() or None
        beam_size = int(transcription_cfg.get("beam_size", 2))
        max_chars = int(transcription_cfg.get("max_chars", 12000))

        model = _get_whisper_model(model_name=model_name, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            str(file_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        text_parts = []
        for seg in segments:
            text = (getattr(seg, "text", "") or "").strip()
            if text:
                text_parts.append(text)

        transcript = " ".join(text_parts).strip()
        if not transcript:
            return None

        if max_chars > 0 and len(transcript) > max_chars:
            transcript = transcript[:max_chars].rstrip() + " ..."

        detected_lang = getattr(info, "language", None)
        if detected_lang:
            return f"[TRANSCRIPT lang={detected_lang}]\n{transcript}"
        return f"[TRANSCRIPT]\n{transcript}"

    except Exception as e:
        print(f"[parser] Whisper transcription failed for {file_path.name}: {e}")
        return None


def parse_media(file_path: pathlib.Path, transcription_cfg: Optional[Dict] = None) -> str:
    """Extract lightweight metadata text for audio/video files."""
    transcription_cfg = transcription_cfg or {}
    suffix = file_path.suffix.lower()
    media_type = "audio" if suffix in SUPPORTED_AUDIO_EXTENSIONS else "video"
    size_bytes = 0
    try:
        size_bytes = file_path.stat().st_size
    except Exception:
        pass

    details: Dict[str, str] = {
        "type": media_type,
        "name": file_path.name,
        "extension": suffix,
        "path": str(file_path),
        "size_bytes": str(size_bytes),
    }

    # Optional WAV duration without extra dependencies
    if suffix == ".wav":
        try:
            import wave
            with wave.open(str(file_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 1
                details["duration_seconds"] = f"{frames / rate:.2f}"
        except Exception:
            pass

    metadata_text = "\n".join(f"{k}: {v}" for k, v in details.items())
    transcript = _transcribe_with_whisper(file_path, transcription_cfg)
    if transcript:
        return f"{metadata_text}\n\n{transcript}"
    return metadata_text


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
    for ext in SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS:
        parsers[ext] = parse_media
    return parsers.get(extension.lower())
