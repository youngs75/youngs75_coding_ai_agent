from __future__ import annotations

from unittest.mock import MagicMock

from deepeval.dataset.golden import Golden


def test_generate_synthetic_dataset_uses_contexts_api(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.md").write_text("SLA content A", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("SLA content B", encoding="utf-8")

    output_path = tmp_path / "synthetic.json"

    class FakeSynthesizer:
        def __init__(self, model):
            self.model = model
            self.called_kwargs = None

        def generate_goldens_from_contexts(self, **kwargs):
            self.called_kwargs = kwargs
            return [
                Golden(
                    input="q1",
                    expected_output="a1",
                    context=["ctx1"],
                    source_file="a.md",
                    synthetic_input_quality=0.9,
                ),
                Golden(
                    input="q2",
                    expected_output="a2",
                    context=["ctx2"],
                    source_file="b.txt",
                    synthetic_input_quality=0.8,
                ),
            ]

    fake_synth = FakeSynthesizer(model=MagicMock())
    monkeypatch.setattr(
        "coding_agent.eval_pipeline.loop1_dataset.synthesizer.get_deepeval_model",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "coding_agent.eval_pipeline.loop1_dataset.synthesizer.Synthesizer",
        lambda model: fake_synth,
    )

    from coding_agent.eval_pipeline.loop1_dataset.synthesizer import (
        generate_synthetic_dataset,
    )

    items = generate_synthetic_dataset(
        corpus_dir=corpus_dir,
        output_path=output_path,
        num_goldens=1,
        max_goldens_per_context=2,
    )

    assert len(items) == 1
    assert output_path.exists()
    assert fake_synth.called_kwargs is not None
    assert "contexts" in fake_synth.called_kwargs
    assert len(fake_synth.called_kwargs["contexts"]) == 2
