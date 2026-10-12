import json
from dataclasses import dataclass
from pathlib import Path

from airgap_rag.evaluation.models import EvaluationReport, GenerationEvaluationReport


class EvaluationReportError(OSError):
    """An evaluation report could not be persisted."""


@dataclass(frozen=True, slots=True)
class ReportPaths:
    json: Path
    markdown: Path


def write_evaluation_reports(
    report: EvaluationReport,
    output_directory: Path,
    *,
    report_name: str | None = None,
) -> ReportPaths:
    safe_name = report_name or f"retrieval-{report.evaluated_at:%Y%m%dT%H%M%SZ}"
    if not safe_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in safe_name
    ):
        raise EvaluationReportError("Report name may contain only letters, digits, '-' and '_'")

    json_path = output_directory / f"{safe_name}.json"
    markdown_path = output_directory / f"{safe_name}.md"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    except OSError as exc:
        raise EvaluationReportError(f"Cannot write reports to {output_directory}: {exc}") from exc
    return ReportPaths(json=json_path, markdown=markdown_path)


def _render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Dataset: `{report.dataset_name}`",
        f"- API: `{report.base_url}`",
        f"- Cases: {report.case_count}",
        f"- Evaluated at: `{report.evaluated_at.isoformat()}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| Hit@{cutoff} | {report.hit_at_k[cutoff]:.4f} |" for cutoff in report.top_k)
    lines.append(f"| MRR@{report.mrr_cutoff} | {report.mrr:.4f} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | First relevant rank | Reciprocal rank |",
            "|---|---:|---:|",
        ]
    )
    for result in report.cases:
        rank = "—" if result.first_relevant_rank is None else str(result.first_relevant_rank)
        lines.append(f"| `{result.case_id}` | {rank} | {result.reciprocal_rank:.4f} |")
    lines.append("")
    return "\n".join(lines)


def write_generation_reports(
    report: GenerationEvaluationReport,
    output_directory: Path,
    *,
    report_name: str | None = None,
) -> ReportPaths:
    safe_name = report_name or f"generation-{report.evaluated_at:%Y%m%dT%H%M%SZ}"
    if not safe_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in safe_name
    ):
        raise EvaluationReportError("Report name may contain only letters, digits, '-' and '_'")
    json_path = output_directory / f"{safe_name}.json"
    markdown_path = output_directory / f"{safe_name}.md"
    lines = [
        "# Generation evaluation",
        "",
        f"- Dataset: `{report.dataset_name}`",
        f"- Cases: {report.case_count}",
        f"- Passed: **{report.passed}**",
        "",
        "| Metric | Value | Threshold |",
        "|---|---:|---:|",
    ]
    for name in ("answer_f1", "citation_precision", "citation_recall", "faithfulness"):
        lines.append(f"| {name} | {getattr(report, name):.4f} | {report.thresholds[name]:.4f} |")
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise EvaluationReportError(f"Cannot write reports to {output_directory}: {exc}") from exc
    return ReportPaths(json=json_path, markdown=markdown_path)
