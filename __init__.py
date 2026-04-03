"""youngs75_a2a — AI Assistant Coding Agent Harness.

A2A 멀티에이전트 프레임워크 + DeepEval 평가 파이프라인 통합 패키지.

구조:
    core/           — 도메인 무관 프레임워크 (BaseGraphAgent, Config, MCP 로더 등)
    a2a/            — A2A 프로토콜 통합 (Executor, Server)
    agents/         — 에이전트 구현체 (SimpleReAct, DeepResearch, Orchestrator, CodingAssistant)
    eval_pipeline/  — DeepEval 평가 파이프라인 (Dataset → Evaluation → Remediation)
    utils/          — 유틸리티 (로깅, 환경변수)
    scripts/        — 파이프라인 실행 스크립트
"""
