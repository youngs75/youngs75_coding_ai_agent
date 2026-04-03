"""eval_pipeline — DeepEval 기반 AI Agent 평가 파이프라인.

Closed-Loop 시스템: Dataset 생성 → Evaluation → Remediation.

구조:
    llm/                — LLM 클라이언트 (OpenRouter, DeepEval 모델)
    loop1_dataset/      — Synthetic/Golden Dataset 생성
    loop2_evaluation/   — 메트릭 평가, 배치 실행, Langfuse 연동
    loop3_remediation/  — 개선안 분석 및 추천
    observability/      — Langfuse 관측성 통합
"""
