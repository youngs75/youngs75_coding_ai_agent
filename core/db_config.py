"""DB 환경변수 설정 클래스.

Pydantic BaseSettings를 사용하여 환경변수에서 DB 설정을 읽는다.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """환경변수 기반 DB 설정 클래스."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    db_host: str = Field(..., alias="DB_HOST")
    db_port: int = Field(..., alias="DB_PORT")
    db_name: str = Field(..., alias="DB_NAME")
