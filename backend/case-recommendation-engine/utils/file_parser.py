import io
import re
from pathlib import Path
from typing import Union


class FileParser:
    """
    Extracts plain text from uploaded legal documents.

    Supports: .pdf, .docx, .doc, .txt
    OOP: class with static/instance methods per file type.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
    MAX_CHARS = 8_000  # Trim to stay within model token limits

    @classmethod
    def parse(cls, file_bytes: bytes, filename: str) -> str:
        """
        Main entry point — detect file type and extract text.

        Args:
            file_bytes: Raw file contents as bytes
            filename:   Original filename (used to detect type)

        Returns:
            Extracted plain text string (max 8000 chars)
        """
        ext = Path(filename).suffix.lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            text = cls._parse_pdf(file_bytes)
        elif ext in {".docx", ".doc"}:
            text = cls._parse_docx(file_bytes)
        elif ext == ".txt":
            text = cls._parse_txt(file_bytes)
        else:
            text = ""

        return cls._clean_and_trim(text)

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n".join(pages)
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        except Exception as e:
            raise ValueError(f"Could not parse PDF: {e}")

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        except Exception as e:
            raise ValueError(f"Could not parse DOCX: {e}")

    @staticmethod
    def _parse_txt(file_bytes: bytes) -> str:
        """Decode plain text file."""
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file — unknown encoding.")

    @classmethod
    def _clean_and_trim(cls, text: str) -> str:
        """Clean extracted text and trim to MAX_CHARS."""
        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove null bytes
        text = text.replace("\x00", "")
        text = text.strip()

        if len(text) > cls.MAX_CHARS:
            text = text[: cls.MAX_CHARS] + "\n\n[Document truncated for analysis]"

        return text

    @staticmethod
    def validate_file_size(file_bytes: bytes, max_mb: int = 10) -> None:
        """Raise ValueError if file exceeds size limit."""
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_mb:
            raise ValueError(
                f"File size {size_mb:.1f} MB exceeds the {max_mb} MB limit."
            )
