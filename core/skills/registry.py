"""스킬 레지스트리 — 스킬 등록, 검색, 활성화를 관리한다.

사용 패턴:
  registry = SkillRegistry(loader=SkillLoader("./skills"))
  registry.discover()                    # L1 메타데이터 일괄 탐색
  ctx = registry.get_context_entries()   # L1 → 프롬프트에 주입
  skill = registry.activate("code_review")  # L2 본문 로드
  skill = registry.activate("code_review", with_refs=True)  # L3 참조 포함
"""

from __future__ import annotations

from youngs75_a2a.core.skills.loader import SkillLoader
from youngs75_a2a.core.skills.schemas import Skill, SkillLevel


# task_type → 검색할 스킬 태그 매핑
TASK_TYPE_TAGS: dict[str, list[str]] = {
    "generate": ["quality"],
    "fix": ["fix", "debug"],
    "refactor": ["refactor"],
    "explain": ["explain"],
    "analyze": ["review", "security"],
    "scaffold": ["scaffold", "framework"],
    "create": ["scaffold", "framework"],
}


class SkillRegistry:
    """스킬 레지스트리 — 3-Level Progressive Loading 관리."""

    def __init__(self, loader: SkillLoader | None = None):
        self._loader = loader
        self._skills: dict[str, Skill] = {}
        self._activation_count: dict[str, int] = {}

    def discover(self) -> list[str]:
        """L1: 모든 스킬 메타데이터를 탐색하여 등록한다.

        Returns:
            등록된 스킬 이름 목록
        """
        if not self._loader:
            return []
        for skill in self._loader.discover():
            if skill.metadata.enabled:
                self._skills[skill.name] = skill
        return list(self._skills.keys())

    def register(self, skill: Skill) -> None:
        """스킬을 수동 등록한다."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """등록된 스킬을 반환한다 (현재 로드 레벨 그대로)."""
        return self._skills.get(name)

    def activate(self, name: str, *, with_refs: bool = False) -> Skill | None:
        """L2/L3: 스킬을 활성화하여 본문(+ 참조)을 로드한다.

        Args:
            name: 스킬 이름
            with_refs: True이면 L3(참조 파일)까지 로드
        """
        skill = self._skills.get(name)
        if skill is None:
            return None

        level = SkillLevel.L3_REFERENCES if with_refs else SkillLevel.L2_BODY

        # 이미 해당 레벨 이상이면 재로드 불필요
        if skill.loaded_level >= level:
            self._track_activation(name)
            return skill

        # 로더로 재로드
        if self._loader:
            loaded = self._loader.load(name, level=level)
            if loaded:
                self._skills[name] = loaded
                self._track_activation(name)
                return loaded

        self._track_activation(name)
        return skill

    def get_context_entries(self) -> list[str]:
        """L1: 모든 활성 스킬의 컨텍스트 주입용 문자열 목록."""
        return [
            skill.as_context_entry()
            for skill in self._skills.values()
            if skill.metadata.enabled
        ]

    def list_skills(self) -> list[Skill]:
        """등록된 모든 스킬 목록."""
        return list(self._skills.values())

    def search_by_tags(self, tags: list[str]) -> list[Skill]:
        """태그로 스킬 검색."""
        if not tags:
            return self.list_skills()
        query_set = set(tags)
        return [
            skill
            for skill in self._skills.values()
            if skill.metadata.enabled and set(skill.metadata.tags) & query_set
        ]

    def auto_activate_for_task(
        self,
        task_type: str,
        framework_hint: str = "",
    ) -> list[str]:
        """task_type에 맞는 스킬을 자동 검색 후 L2 활성화한다.

        Args:
            task_type: parse 결과의 task_type (generate, fix, refactor, explain, analyze, scaffold)
            framework_hint: scaffold일 때 프레임워크 힌트 (예: "flask_vue").
                           지정하면 해당 이름의 스킬만 활성화한다.

        Returns:
            활성화된 스킬 이름 목록
        """
        tags = TASK_TYPE_TAGS.get(task_type, [])
        if not tags:
            return []

        matched = self.search_by_tags(tags)

        # framework_hint가 있으면 해당 스킬만 필터링 (4개 전부 활성화 방지)
        if framework_hint:
            hint_lower = framework_hint.lower().replace("-", "_")
            filtered = [s for s in matched if s.name == hint_lower]
            if filtered:
                matched = filtered

        activated: list[str] = []
        for skill in matched:
            result = self.activate(skill.name)
            if result:
                activated.append(result.name)
        return activated

    def get_active_skill_bodies(self) -> list[str]:
        """L2 이상 활성화된 스킬의 본문을 컨텍스트 주입용 문자열로 반환한다.

        활성화되지 않은 스킬(L1 메타데이터만)은 제외한다.
        """
        entries: list[str] = []
        for skill in self._skills.values():
            if skill.loaded_level >= SkillLevel.L2_BODY and skill.body:
                entries.append(f"### 스킬: {skill.name}\n{skill.body}")
        return entries

    @property
    def activation_stats(self) -> dict[str, int]:
        """스킬별 활성화 횟수 통계."""
        return dict(self._activation_count)

    def _track_activation(self, name: str) -> None:
        self._activation_count[name] = self._activation_count.get(name, 0) + 1
