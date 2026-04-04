"""LLM 기반 피드백 보강 모듈 (Loop 1 - Step 3b).

Human Review에서 받은 피드백을 LLM을 활용하여 expected_output에 반영합니다.
리뷰어가 "좀 더 구체적인 예시를 추가해주세요" 같은 피드백을 남기면,
LLM이 기존 답변을 개선하여 더 나은 Golden Dataset을 만듭니다.

작동 흐름:
    1. feedback 필드가 비어있지 않은 항목을 찾음
    2. 각 항목에 대해 LLM에게 피드백 기반 개선을 요청
    3. 개선된 expected_output으로 교체
    4. augmented=True 플래그와 개선 노트를 추가

사용 예시:
    from youngs75_a2a.eval_pipeline.loop1_dataset.feedback_augmenter import augment_with_feedback
    improved_items = augment_with_feedback(golden_items)
"""

from __future__ import annotations

import json

from youngs75_a2a.eval_pipeline.llm.openrouter import get_openrouter_client
from youngs75_a2a.eval_pipeline.loop1_dataset.prompts import FEEDBACK_AUGMENT_PROMPT
from youngs75_a2a.eval_pipeline.settings import get_settings


def augment_with_feedback(items: list[dict]) -> list[dict]:
    """Human Review 피드백을 LLM으로 보강하여 expected_output을 개선합니다.

    feedback 필드가 비어있지 않은 항목에 대해서만 LLM 보강을 수행합니다.
    피드백이 없는 항목은 그대로 통과합니다.

    보강 실패 시(API 오류 등) 원본 데이터를 유지하고
    augmented=False로 표시합니다.

    Args:
        items: Golden Dataset 항목 리스트. 각 항목에 feedback 필드가 있어야 함

    Returns:
        보강된 항목 리스트. 각 항목에 다음 필드가 추가됨:
            - augmented: bool (LLM 보강 성공 여부)
            - augmentation_notes: str (개선 내용 설명)
    """
    settings = get_settings()
    client = get_openrouter_client()
    augmented = []

    for item in items:
        feedback = item.get("feedback", "").strip()

        # 피드백이 없으면 보강 없이 그대로 통과
        if not feedback:
            augmented.append(item)
            continue

        # context 리스트를 문자열로 변환 (프롬프트에 삽입하기 위해)
        context_str = item.get("context", [])
        if isinstance(context_str, list):
            context_str = "; ".join(context_str)

        # 프롬프트 구성: 현재 데이터 + 피드백을 LLM에게 전달
        prompt = FEEDBACK_AUGMENT_PROMPT.format(
            input=item.get("input", ""),
            expected_output=item.get("expected_output", ""),
            context=context_str,
            feedback=feedback,
        )

        try:
            # LLM 호출: 피드백을 반영한 개선 답변 생성
            # Settings의 openrouter_model_name은 DEFAULT_MODEL 환경변수에서 로드
            model_name = settings.openrouter_model_name or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 결정론적 출력
            )
            content = response.choices[0].message.content or ""
            result = json.loads(content)

            # 개선된 답변으로 교체
            improved = {**item}
            improved["expected_output"] = result.get(
                "improved_expected_output", item["expected_output"]
            )
            improved["augmentation_notes"] = result.get("improvement_notes", "")
            improved["augmented"] = True
            augmented.append(improved)
        except Exception as exc:
            # API 오류 시 원본 유지, 실패 표시
            print(
                f"[WARN] Feedback augmentation failed for '{item.get('id', 'unknown')}': {exc}"
            )
            item_copy = {
                **item,
                "augmented": False,
                "augmentation_notes": "LLM augmentation failed",
            }
            augmented.append(item_copy)

    return augmented
