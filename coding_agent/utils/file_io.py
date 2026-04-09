"""파일 읽기/쓰기를 안전하게 처리하는 유틸리티."""

from __future__ import annotations

from pathlib import Path


class FileIOError(Exception):
    """파일 입출력 처리 중 발생하는 예외."""


class SafeFileIO:
    """컨텍스트 매니저를 사용해 파일 읽기/쓰기를 안전하게 처리한다."""

    @staticmethod
    def read_text(path: str | Path, encoding: str = "utf-8") -> str:
        """텍스트 파일을 안전하게 읽어 내용을 반환한다."""
        file_path = Path(path)
        try:
            with file_path.open("r", encoding=encoding) as file:
                return file.read()
        except OSError as exc:
            raise FileIOError(f"파일 읽기에 실패했습니다: {file_path}") from exc

    @staticmethod
    def write_text(
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Path:
        """텍스트 파일에 내용을 안전하게 쓴다."""
        file_path = Path(path)
        try:
            if create_parents:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding=encoding) as file:
                file.write(content)
            return file_path
        except OSError as exc:
            raise FileIOError(f"파일 쓰기에 실패했습니다: {file_path}") from exc

    @staticmethod
    def append_text(
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Path:
        """텍스트 파일 끝에 내용을 안전하게 추가한다."""
        file_path = Path(path)
        try:
            if create_parents:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", encoding=encoding) as file:
                file.write(content)
            return file_path
        except OSError as exc:
            raise FileIOError(f"파일 추가 쓰기에 실패했습니다: {file_path}") from exc
