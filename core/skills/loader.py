"""스킬 로더 — YAML/JSON 파일에서 스킬을 로드한다.

스킬 파일 포맷 (YAML):
  name: code_review
  description: 코드 리뷰 수행
  tags: [review, quality]
  version: "1.0.0"
  body: |
    당신은 코드 리뷰어입니다. 다음 코드를 검토하세요...
  references:
    - path: ./templates/review_checklist.md
      description: 리뷰 체크리스트
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coding_agent.core.skills.schemas import (
    Skill,
    SkillLevel,
    SkillMetadata,
    SkillReference,
)


def _parse_skill_dict(data: dict[str, Any], source_path: Path | None = None) -> Skill:
    """딕셔너리에서 Skill 객체를 생성한다."""
    metadata = SkillMetadata(
        name=data["name"],
        description=data.get("description", ""),
        tags=data.get("tags", []),
        version=data.get("version", "1.0.0"),
        enabled=data.get("enabled", True),
    )
    references = [
        SkillReference(path=ref["path"], description=ref.get("description", ""))
        for ref in data.get("references", [])
    ]
    return Skill(
        metadata=metadata,
        body=data.get("body"),
        references=references,
        source_path=source_path,
        extra=data.get("extra", {}),
    )


class SkillLoader:
    """파일 시스템에서 스킬을 로드한다.

    Args:
        skills_dir: 스킬 파일이 위치한 디렉토리
    """

    def __init__(self, skills_dir: Path | str):
        self._dir = Path(skills_dir)

    def discover(self) -> list[Skill]:
        """디렉토리에서 모든 스킬을 L1(메타데이터)으로 탐색한다."""
        skills: list[Skill] = []
        if not self._dir.exists():
            return skills
        for path in sorted(self._dir.iterdir()):
            if path.suffix in (".yaml", ".yml", ".json"):
                skill = self._load_file(path, level=SkillLevel.L1_METADATA)
                if skill:
                    skills.append(skill)
        return skills

    def load(self, name: str, level: SkillLevel = SkillLevel.L2_BODY) -> Skill | None:
        """이름으로 스킬을 지정 레벨까지 로드한다."""
        for ext in (".yaml", ".yml", ".json"):
            path = self._dir / f"{name}{ext}"
            if path.exists():
                return self._load_file(path, level=level)
        return None

    def load_references(self, skill: Skill) -> Skill:
        """L3: 스킬의 참조 파일 내용을 로드한다."""
        base_dir = skill.source_path.parent if skill.source_path else self._dir
        updated_refs: list[SkillReference] = []
        for ref in skill.references:
            ref_path = base_dir / ref.path
            content = None
            if ref_path.exists():
                content = ref_path.read_text(encoding="utf-8")
            updated_refs.append(
                SkillReference(
                    path=ref.path, description=ref.description, content=content
                )
            )
        return skill.model_copy(update={"references": updated_refs})

    def _load_file(self, path: Path, level: SkillLevel) -> Skill | None:
        """파일에서 스킬을 로드한다."""
        try:
            raw = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                data = json.loads(raw)
            else:
                data = self._parse_yaml(raw)
            skill = _parse_skill_dict(data, source_path=path)

            # L1만 요청 시 body 제거
            if level == SkillLevel.L1_METADATA:
                skill = skill.model_copy(update={"body": None, "references": []})
            elif level >= SkillLevel.L3_REFERENCES:
                skill = self.load_references(skill)

            return skill
        except Exception:
            return None

    @staticmethod
    def _parse_yaml(raw: str) -> dict[str, Any]:
        """간단한 YAML 파서 (PyYAML 의존 없이 기본 키-값 파싱).

        복잡한 YAML은 PyYAML로 폴백한다.
        """
        try:
            import yaml

            return yaml.safe_load(raw)
        except ImportError:
            # PyYAML 없으면 JSON 폴백 시도
            return json.loads(raw)
