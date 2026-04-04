from __future__ import annotations

import json


class TestCSVExportImport:
    """CSV 내보내기/가져오기 순수 로직 테스트."""

    def test_export_to_review_csv(self, tmp_data_dir):
        from youngs75_a2a.eval_pipeline.loop1_dataset.csv_exporter import (
            export_to_review_csv,
        )

        synthetic_data = [
            {
                "id": "s001",
                "input": "테스트 질문",
                "expected_output": "테스트 답변",
                "context": ["ctx1", "ctx2"],
                "source_file": "test.md",
                "synthetic_input_quality": 0.8,
            }
        ]
        synthetic_path = tmp_data_dir / "synthetic" / "test.json"
        with open(synthetic_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_data, f)

        output_path = tmp_data_dir / "review" / "test.csv"
        result = export_to_review_csv(synthetic_path, output_path)

        assert result.exists()
        import pandas as pd

        df = pd.read_csv(result, encoding="utf-8-sig")
        assert len(df) == 1
        assert df.iloc[0]["input"] == "테스트 질문"
        assert df.iloc[0]["context"] == "ctx1;ctx2"
        assert pd.isna(df.iloc[0]["approved"])

    def test_import_reviewed_csv(self, tmp_data_dir):
        import pandas as pd

        from youngs75_a2a.eval_pipeline.loop1_dataset.csv_importer import (
            import_reviewed_csv,
        )

        csv_data = pd.DataFrame(
            [
                {
                    "id": "s001",
                    "input": "승인된 질문",
                    "expected_output": "승인된 답변",
                    "context": "ctx1;ctx2",
                    "source_file": "test.md",
                    "synthetic_input_quality": 0.8,
                    "approved": "True",
                    "feedback": "좋은 질문",
                    "reviewer": "tester",
                },
                {
                    "id": "s002",
                    "input": "거절된 질문",
                    "expected_output": "거절된 답변",
                    "context": "ctx3",
                    "source_file": "test.md",
                    "synthetic_input_quality": 0.3,
                    "approved": "False",
                    "feedback": "관련 없음",
                    "reviewer": "tester",
                },
            ]
        )
        csv_path = tmp_data_dir / "review" / "reviewed.csv"
        csv_data.to_csv(csv_path, index=False, encoding="utf-8-sig")

        golden_path = tmp_data_dir / "golden" / "golden.json"
        items = import_reviewed_csv(csv_path, golden_path, only_approved=True)

        assert len(items) == 1
        assert items[0]["input"] == "승인된 질문"
        assert items[0]["context"] == ["ctx1", "ctx2"]
        assert items[0]["approved"] is True

    def test_import_all_items(self, tmp_data_dir):
        import pandas as pd

        from youngs75_a2a.eval_pipeline.loop1_dataset.csv_importer import (
            import_reviewed_csv,
        )

        csv_data = pd.DataFrame(
            [
                {
                    "id": "s001",
                    "input": "q1",
                    "expected_output": "a1",
                    "context": "c1",
                    "source_file": "t.md",
                    "synthetic_input_quality": 0.8,
                    "approved": "True",
                    "feedback": "",
                    "reviewer": "t",
                },
                {
                    "id": "s002",
                    "input": "q2",
                    "expected_output": "a2",
                    "context": "c2",
                    "source_file": "t.md",
                    "synthetic_input_quality": 0.5,
                    "approved": "False",
                    "feedback": "",
                    "reviewer": "t",
                },
            ]
        )
        csv_path = tmp_data_dir / "review" / "all.csv"
        csv_data.to_csv(csv_path, index=False, encoding="utf-8-sig")

        golden_path = tmp_data_dir / "golden" / "all.json"
        items = import_reviewed_csv(csv_path, golden_path, only_approved=False)

        assert len(items) == 2
