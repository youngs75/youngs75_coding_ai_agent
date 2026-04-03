"""CLI 설정."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class CLIConfig(BaseModel):
    """대화형 CLI 설정."""

    default_agent: str = Field(
        default_factory=lambda: os.getenv("CLI_DEFAULT_AGENT", "coding_assistant"),
    )
    stream_output: bool = True
    history_file: str = Field(
        default_factory=lambda: os.getenv("CLI_HISTORY_FILE", ".cli_history"),
    )
    max_history: int = 1000
    theme: str = Field(
        default_factory=lambda: os.getenv("CLI_THEME", "monokai"),
    )
    skills_dir: str | None = Field(
        default_factory=lambda: os.getenv("SKILLS_DIR"),
        description="스킬 파일 디렉토리 경로",
    )
