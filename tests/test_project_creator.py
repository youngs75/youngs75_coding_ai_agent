"""Project Creator MCP 서버 + 프레임워크 스킬 통합 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.core.skills.loader import SkillLoader
from coding_agent.core.skills.registry import TASK_TYPE_TAGS, SkillRegistry


# ── Project Creator MCP 서버 도구 테스트 ──


class TestProjectCreatorTools:
    """MCP 서버 도구 함수를 직접 임포트하여 테스트."""

    @pytest.fixture(autouse=True)
    def _patch_workspace(self, tmp_path, monkeypatch):
        """workspace를 tmp_path로 패치."""
        monkeypatch.setattr(
            "coding_agent.mcp_servers.code_tools.server._WORKSPACE", str(tmp_path)
        )
        self.workspace = tmp_path

    def test_list_templates(self):
        from coding_agent.mcp_servers.code_tools.server import list_templates

        result = list_templates()
        assert "flask-vue" in result
        assert "react-express" in result
        assert "fastapi-react" in result
        assert "django-htmx" in result

    def test_get_template_info_valid(self):
        from coding_agent.mcp_servers.code_tools.server import get_template_info

        result = get_template_info("flask-vue")
        assert "Flask" in result
        assert "Vue" in result
        assert "디렉토리 구조" in result

    def test_get_template_info_invalid(self):
        from coding_agent.mcp_servers.code_tools.server import get_template_info

        result = get_template_info("nonexistent")
        assert "Error" in result
        assert "flask-vue" in result  # 사용 가능한 목록 표시

    def test_create_project_flask_vue(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("flask-vue", "my-app")
        assert "생성 완료" in result
        assert (self.workspace / "my-app" / "backend" / "app" / "__init__.py").exists()
        assert (self.workspace / "my-app" / "frontend" / "package.json").exists()
        assert (self.workspace / "my-app" / "README.md").exists()

    def test_create_project_react_express(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("react-express", "web-app")
        assert "생성 완료" in result
        assert (self.workspace / "web-app" / "server" / "index.js").exists()
        assert (self.workspace / "web-app" / "client" / "package.json").exists()

    def test_create_project_fastapi_react(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("fastapi-react", "api-app")
        assert "생성 완료" in result
        assert (self.workspace / "api-app" / "backend" / "app" / "main.py").exists()
        assert (self.workspace / "api-app" / "frontend" / "vite.config.js").exists()

    def test_create_project_django_htmx(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("django-htmx", "cms-app")
        assert "생성 완료" in result
        assert (self.workspace / "cms-app" / "manage.py").exists()
        assert (self.workspace / "cms-app" / "config" / "settings.py").exists()
        assert (self.workspace / "cms-app" / "templates" / "base.html").exists()

    def test_create_project_template_variable_substitution(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        create_project("flask-vue", "hello-world")
        readme = (self.workspace / "hello-world" / "README.md").read_text()
        assert "hello-world" in readme
        assert "{project_name}" not in readme

    def test_create_project_invalid_template(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("nonexistent", "app")
        assert "Error" in result

    def test_create_project_duplicate_directory(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        create_project("flask-vue", "dup-app")
        result = create_project("flask-vue", "dup-app")
        assert "Error" in result
        assert "이미 존재" in result

    def test_create_project_invalid_options_json(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        result = create_project("flask-vue", "bad-opts", options="{invalid}")
        assert "Error" in result
        assert "JSON" in result

    def test_safe_path_blocks_traversal(self):
        from coding_agent.mcp_servers.code_tools.server import create_project

        with pytest.raises(ValueError, match="접근 거부"):
            create_project("flask-vue", "../escape-attempt")


# ── 프레임워크 스킬 로드 테스트 ──


class TestFrameworkSkills:
    """data/skills/의 프레임워크 스킬 YAML 로드 및 활성화 테스트."""

    SKILLS_DIR = Path(__file__).resolve().parent.parent / "data" / "skills"

    @pytest.fixture()
    def registry(self) -> SkillRegistry:
        loader = SkillLoader(self.SKILLS_DIR)
        reg = SkillRegistry(loader=loader)
        reg.discover()
        return reg

    def test_framework_skills_discovered(self, registry):
        """4개 프레임워크 스킬이 발견되어야 한다."""
        names = {s.name for s in registry.list_skills()}
        assert "flask_vue" in names
        assert "react_express" in names
        assert "fastapi_react" in names
        assert "django_htmx" in names

    def test_flask_vue_skill_activate(self, registry):
        skill = registry.activate("flask_vue")
        assert skill is not None
        assert skill.body is not None
        assert "App Factory" in skill.body or "create_app" in skill.body

    def test_react_express_skill_activate(self, registry):
        skill = registry.activate("react_express")
        assert skill is not None
        assert skill.body is not None
        assert "Express" in skill.body

    def test_fastapi_react_skill_activate(self, registry):
        skill = registry.activate("fastapi_react")
        assert skill is not None
        assert skill.body is not None
        assert "FastAPI" in skill.body

    def test_django_htmx_skill_activate(self, registry):
        skill = registry.activate("django_htmx")
        assert skill is not None
        assert skill.body is not None
        assert "HTMX" in skill.body or "htmx" in skill.body

    def test_scaffold_tag_search(self, registry):
        """scaffold 태그로 프레임워크 스킬을 검색할 수 있어야 한다."""
        results = registry.search_by_tags(["scaffold"])
        names = {s.name for s in results}
        assert len(names) >= 4
        assert "flask_vue" in names

    def test_framework_tag_search(self, registry):
        """framework 태그로 프레임워크 스킬을 검색할 수 있어야 한다."""
        results = registry.search_by_tags(["framework"])
        assert len(results) >= 4


# ── TASK_TYPE_TAGS 매핑 테스트 ──


class TestTaskTypeTags:
    """scaffold/create task_type이 TASK_TYPE_TAGS에 등록되어야 한다."""

    def test_scaffold_mapping_exists(self):
        assert "scaffold" in TASK_TYPE_TAGS
        assert "scaffold" in TASK_TYPE_TAGS["scaffold"]
        assert "framework" in TASK_TYPE_TAGS["scaffold"]

    def test_create_mapping_exists(self):
        assert "create" in TASK_TYPE_TAGS
        assert "scaffold" in TASK_TYPE_TAGS["create"]
        assert "framework" in TASK_TYPE_TAGS["create"]

    def test_auto_activate_scaffold(self, tmp_path):
        """scaffold task_type → 프레임워크 스킬 자동 활성화."""
        (tmp_path / "flask_vue.json").write_text(
            json.dumps({
                "name": "flask_vue",
                "description": "Flask + Vue",
                "tags": ["scaffold", "framework", "flask", "vue"],
                "body": "Flask + Vue 가이드",
            }),
            encoding="utf-8",
        )
        loader = SkillLoader(tmp_path)
        reg = SkillRegistry(loader=loader)
        reg.discover()

        activated = reg.auto_activate_for_task("scaffold")
        assert "flask_vue" in activated

    def test_auto_activate_create(self, tmp_path):
        """create task_type → 프레임워크 스킬 자동 활성화."""
        (tmp_path / "react_express.json").write_text(
            json.dumps({
                "name": "react_express",
                "description": "React + Express",
                "tags": ["scaffold", "framework", "react", "express"],
                "body": "React + Express 가이드",
            }),
            encoding="utf-8",
        )
        loader = SkillLoader(tmp_path)
        reg = SkillRegistry(loader=loader)
        reg.discover()

        activated = reg.auto_activate_for_task("create")
        assert "react_express" in activated
