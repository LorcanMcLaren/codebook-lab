from __future__ import annotations

from codebook_lab.metrics import format_metrics_summary


class TestFormatMetricsSummary:
    def test_classification_and_textbox_summary(self):
        summary = format_metrics_summary(
            {
                "Policy Sentiment_Direction": {
                    "annotation_type": "dropdown",
                    "percentage_agreement": 0.2,
                    "accuracy": 0.25,
                    "f1": 0.3,
                    "cohen_kappa": 0.1,
                    "krippendorff_alpha": 0.05,
                },
                "Policy Sentiment_Evidence": {
                    "annotation_type": "textbox",
                    "percentage_agreement": 0.0,
                    "norm_levenshtein": 0.4,
                    "bleu": 0.2,
                    "rougeL_f": 0.5,
                    "cosine_similarity": 0.7,
                    "bertscore_f1": 0.8,
                },
            },
            total_inference_time=12.345,
            avg_inference_time=1.234,
            input_chars=4321,
            output_chars=987,
            energy_consumed=0.0123,
            emissions=0.0045,
        )

        assert "Run Summary" in summary
        assert "Performance" in summary
        assert "Policy Sentiment_Direction [dropdown]" in summary
        assert "accuracy=0.250" in summary
        assert "Policy Sentiment_Evidence [textbox]" in summary
        assert "bertscore_f1=0.800" in summary
        assert "Efficiency" in summary
        assert "Total inference time (s): 12.345" in summary
        assert "Output characters: 987" in summary

    def test_empty_summary(self):
        assert format_metrics_summary({}) == "No metrics were produced."
