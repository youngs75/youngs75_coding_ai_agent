"""Loop 1 Dataset 프롬프트 모듈.

Loop 1 (Dataset 생성/보강) 과정에서 LLM에 전달하는 모든 프롬프트를
한 곳에서 관리합니다. 인라인 프롬프트 대신 이 파일에서 import하여 사용합니다.

프롬프트 목록:
    - FEEDBACK_AUGMENT_PROMPT: Human Review 피드백을 반영하여
      expected_output을 개선하는 프롬프트 템플릿

사용 예시:
    from youngs75_a2a.eval_pipeline.loop1_dataset.prompts import FEEDBACK_AUGMENT_PROMPT

    prompt = FEEDBACK_AUGMENT_PROMPT.format(
        input="사용자 질문",
        expected_output="기존 답변",
        context="관련 컨텍스트",
        feedback="리뷰어 피드백",
    )
"""

from __future__ import annotations

# ── 피드백 보강 프롬프트 ──────────────────────────────────────
# feedback_augmenter.py에서 사용합니다.
# {input}, {expected_output}, {context}, {feedback} 자리에 실제 값이 포맷팅됩니다.
# 이중 중괄호 {{, }}는 Python str.format()에서 리터럴 중괄호를 표현합니다.
FEEDBACK_AUGMENT_PROMPT = """\
<task>
Improve the expected_output of a test case based on human reviewer feedback.
The improved answer will be used as the golden standard for future evaluations,
so accuracy and completeness are critical.
</task>

<test_case>
- Input: {input}
- Current expected_output: {expected_output}
- Context: {context}
- Human feedback: {feedback}
</test_case>

<instructions>
1. Read the human feedback carefully and identify what needs to change.
2. Revise the expected_output to address every point in the feedback.
3. Preserve all accurate information from the original expected_output.
4. Ensure the revised answer is fully supported by the provided context.
</instructions>

<output_format>
Return a JSON object with exactly these fields:
{{
  "improved_expected_output": "the revised answer addressing all feedback",
  "improvement_notes": "what was changed and why, referencing specific feedback points"
}}
</output_format>
"""
