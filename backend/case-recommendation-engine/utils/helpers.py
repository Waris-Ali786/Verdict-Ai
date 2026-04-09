import time
import logging
import functools
from typing import Any, Callable
from datetime import datetime


# ──────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a formatted logger for any module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Model trained successfully")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ──────────────────────────────────────────────────
# Decorators (Python OOP / functional programming)
# ──────────────────────────────────────────────────

def timer(func: Callable) -> Callable:
    """
    Decorator: measure and print execution time of any function.

    Usage:
        @timer
        def train_model(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱  {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator factory: retry a function on exception.

    Usage:
        @retry(max_attempts=3, delay=2.0)
        def call_api(): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"  Attempt {attempt}/{max_attempts} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator


# ──────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────

def format_score(score: float, decimals: int = 1) -> str:
    """Format a 0–100 relevance score with a visual bar."""
    filled = int(score / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {score:.{decimals}f}%"


def truncate(text: str, max_len: int = 150, suffix: str = "...") -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)].rstrip() + suffix


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024**2:.1f} MB"


def risk_badge(risk_level: str) -> str:
    """Return an ASCII badge for a risk level."""
    badges = {
        "high":   "🔴 HIGH",
        "medium": "🟡 MEDIUM",
        "low":    "🟢 LOW",
    }
    return badges.get(risk_level.lower(), "⚪ UNKNOWN")


def outcome_badge(outcome: str) -> str:
    """Return an ASCII badge for a case outcome."""
    badges = {
        "guilty":    "⚖️  GUILTY",
        "acquitted": "✅ ACQUITTED",
        "settled":   "🤝 SETTLED",
        "pending":   "⏳ PENDING",
        "dismissed": "❌ DISMISSED",
    }
    return badges.get(outcome.lower(), f"❓ {outcome.upper()}")


# ──────────────────────────────────────────────────
# Text validation
# ──────────────────────────────────────────────────

def validate_case_text(text: str, min_length: int = 20) -> tuple[bool, str]:
    """
    Validate that case text is suitable for analysis.
    Returns (is_valid: bool, error_message: str).
    """
    if not text or not text.strip():
        return False, "Case text cannot be empty."
    if len(text.strip()) < min_length:
        return False, f"Case text too short. Minimum {min_length} characters required."
    if len(text) > 50_000:
        return False, "Case text too long. Maximum 50,000 characters."
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Remove unsafe characters from filenames."""
    import re
    # Keep only alphanumeric, dots, dashes, underscores
    safe = re.sub(r"[^\w.\-]", "_", filename)
    return safe[:255]  # Max filename length


# ──────────────────────────────────────────────────
# JSON serialization helper
# ──────────────────────────────────────────────────

def safe_json(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects.
    Handles: numpy types, datetime, dataclasses.
    """
    import numpy as np
    import dataclasses

    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif dataclasses.is_dataclass(obj):
        return safe_json(dataclasses.asdict(obj))
    return obj


# ──────────────────────────────────────────────────
# Progress printer
# ──────────────────────────────────────────────────

class ProgressPrinter:
    """
    Simple terminal progress tracker for multi-step pipelines.
    Demonstrates: OOP, context managers (__enter__/__exit__).
    """

    def __init__(self, total_steps: int, title: str = "Processing"):
        self.total = total_steps
        self.current = 0
        self.title = title
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        print(f"\n{self.title}")
        print("─" * 40)
        return self

    def step(self, description: str):
        self.current += 1
        pct = self.current / self.total * 100
        bar = "█" * self.current + "░" * (self.total - self.current)
        print(f"  [{bar}] {pct:.0f}% — {description}")

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        print(f"─ Completed in {elapsed:.2f}s " + "─" * 20 + "\n")
