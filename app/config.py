from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    pass
    


@dataclass(frozen=True)
class Settings:
    edgar_user_agent: str
    data_dir: Path
    companies_path: Path
    ontology_path: Path
    max_extraction_calls: int
    log_level: str

    def require_user_agent(self) -> str:
        if "@" not in self.edgar_user_agent:
            raise ConfigError(
                "EDGAR_USER_AGENT must contain your email, "
                "for example: Your Name you@example.com"
            )
        return self.edgar_user_agent


def _whole_number(environ: Mapping[str, str], key: str, default: str) -> int:
    raw = environ.get(key, default).strip()
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a whole number, got {raw!r}") from None


def load_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    """Read settings from `environ`. By default: the .env file plus the real environment."""
    if environ is None:
        from dotenv import load_dotenv

        load_dotenv()
        environ = os.environ

    max_calls = _whole_number(environ, "MAX_EXTRACTION_CALLS", "0")
    if max_calls < 0:
        raise ConfigError("MAX_EXTRACTION_CALLS must be 0 or more")

    level = environ.get("LOG_LEVEL", "INFO").strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")

    return Settings(
        edgar_user_agent=environ.get("EDGAR_USER_AGENT", "").strip(),
        data_dir=Path(environ.get("DATA_DIR", "data")),
        companies_path=Path(environ.get("COMPANIES_PATH", "configs/companies.json")),
        ontology_path=Path(environ.get("ONTOLOGY_PATH", "configs/ontology.json")),
        max_extraction_calls=max_calls,
        log_level=level,
    )
