"""Skills 시스템 유닛 테스트."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from youngs75_a2a.core.skills.loader import SkillLoader
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.skills.schemas import Skill, SkillLevel, SkillMetadata, SkillReference


# ── SkillMetadata / Skill ──


class TestSkillSchemas:
    def test_metadata_defaults(self):
        m = SkillMetadata(name="test", description="desc")
        assert m.enabled is True
        assert m.version == "1.0.0"
        assert m.tags == []

    def test_skill_loaded_level_l1(self):
        skill = Skill(metadata=SkillMetadata(name="s", description="d"))
        assert skill.loaded_level == SkillLevel.L1_METADATA

    def test_skill_loaded_level_l2(self):
        skill = Skill(metadata=SkillMetadata(name="s", description="d"), body="prompt text")
        assert skill.loaded_level == SkillLevel.L2_BODY

    def test_skill_loaded_level_l3(self):
        refs = [SkillReference(path="f.md", content="loaded")]
        skill = Skill(metadata=SkillMetadata(name="s", description="d"), body="x", references=refs)
        assert skill.loaded_level == SkillLevel.L3_REFERENCES

    def test_as_context_entry(self):
        skill = Skill(
            metadata=SkillMetadata(name="review", description="코드 리뷰", tags=["quality"]),
        )
        entry = skill.as_context_entry()
        assert "review" in entry
        assert "코드 리뷰" in entry
        assert "quality" in entry

    def test_name_property(self):
        skill = Skill(metadata=SkillMetadata(name="my_skill", description="d"))
        assert skill.name == "my_skill"


# ── SkillLoader ──


class TestSkillLoader:
    @pytest.fixture()
    def skills_dir(self, tmp_path: Path) -> Path:
        # JSON 스킬 파일
        (tmp_path / "code_review.json").write_text(
            json.dumps({
                "name": "code_review",
                "description": "코드 리뷰 수행",
                "tags": ["review", "quality"],
                "body": "당신은 코드 리뷰어입니다.",
                "references": [
                    {"path": "checklist.md", "description": "체크리스트"},
                ],
            }),
            encoding="utf-8",
        )
        # 참조 파일
        (tmp_path / "checklist.md").write_text("- [ ] 타입 힌트\n- [ ] 에러 처리", encoding="utf-8")

        # 두 번째 스킬
        (tmp_path / "test_gen.json").write_text(
            json.dumps({
                "name": "test_gen",
                "description": "테스트 코드 생성",
                "tags": ["testing"],
                "body": "테스트를 생성하세요.",
            }),
            encoding="utf-8",
        )

        # 비활성 스킬
        (tmp_path / "disabled.json").write_text(
            json.dumps({
                "name": "disabled_skill",
                "description": "비활성",
                "enabled": False,
            }),
            encoding="utf-8",
        )

        return tmp_path

    def test_discover_l1(self, skills_dir):
        loader = SkillLoader(skills_dir)
        skills = loader.discover()
        assert len(skills) == 3
        # L1이므로 body는 None
        for skill in skills:
            assert skill.body is None

    def test_load_l2(self, skills_dir):
        loader = SkillLoader(skills_dir)
        skill = loader.load("code_review", level=SkillLevel.L2_BODY)
        assert skill is not None
        assert skill.body == "당신은 코드 리뷰어입니다."
        assert skill.loaded_level == SkillLevel.L2_BODY

    def test_load_l3_with_references(self, skills_dir):
        loader = SkillLoader(skills_dir)
        skill = loader.load("code_review", level=SkillLevel.L3_REFERENCES)
        assert skill is not None
        assert skill.references[0].content is not None
        assert "타입 힌트" in skill.references[0].content
        assert skill.loaded_level == SkillLevel.L3_REFERENCES

    def test_load_nonexistent(self, skills_dir):
        loader = SkillLoader(skills_dir)
        assert loader.load("nonexistent") is None

    def test_empty_dir(self, tmp_path):
        loader = SkillLoader(tmp_path)
        assert loader.discover() == []

    def test_nonexistent_dir(self):
        loader = SkillLoader("/nonexistent/path")
        assert loader.discover() == []


# ── SkillRegistry ──


class TestSkillRegistry:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> SkillRegistry:
        (tmp_path / "alpha.json").write_text(
            json.dumps({
                "name": "alpha",
                "description": "Alpha 스킬",
                "tags": ["a"],
                "body": "Alpha body",
            }),
            encoding="utf-8",
        )
        (tmp_path / "beta.json").write_text(
            json.dumps({
                "name": "beta",
                "description": "Beta 스킬",
                "tags": ["b"],
                "body": "Beta body",
            }),
            encoding="utf-8",
        )
        loader = SkillLoader(tmp_path)
        reg = SkillRegistry(loader=loader)
        reg.discover()
        return reg

    def test_discover(self, registry):
        names = [s.name for s in registry.list_skills()]
        assert "alpha" in names
        assert "beta" in names

    def test_get(self, registry):
        skill = registry.get("alpha")
        assert skill is not None
        assert skill.name == "alpha"

    def test_activate_l2(self, registry):
        skill = registry.activate("alpha")
        assert skill is not None
        assert skill.body == "Alpha body"
        assert skill.loaded_level >= SkillLevel.L2_BODY

    def test_context_entries(self, registry):
        entries = registry.get_context_entries()
        assert len(entries) == 2
        assert any("Alpha" in e for e in entries)

    def test_search_by_tags(self, registry):
        results = registry.search_by_tags(["a"])
        assert len(results) == 1
        assert results[0].name == "alpha"

    def test_activation_stats(self, registry):
        registry.activate("alpha")
        registry.activate("alpha")
        registry.activate("beta")
        stats = registry.activation_stats
        assert stats["alpha"] == 2
        assert stats["beta"] == 1

    def test_manual_register(self):
        reg = SkillRegistry()
        skill = Skill(metadata=SkillMetadata(name="manual", description="수동 등록"), body="content")
        reg.register(skill)
        assert reg.get("manual") is not None
