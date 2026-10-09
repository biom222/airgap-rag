import json
from pathlib import Path

from pydantic import ValidationError

from airgap_rag.evaluation.models import EvaluationCase


class EvaluationDatasetError(ValueError):
    """The evaluation dataset is missing or does not match the JSONL contract."""


def load_evaluation_dataset(path: Path) -> tuple[EvaluationCase, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDatasetError(f"Cannot read dataset {path}: {exc}") from exc

    cases: list[EvaluationCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_case = json.loads(line)
            case = EvaluationCase.model_validate(raw_case)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvaluationDatasetError(
                f"Invalid dataset record at {path}:{line_number}: {exc}"
            ) from exc
        if case.case_id in case_ids:
            raise EvaluationDatasetError(
                f"Duplicate case_id {case.case_id!r} at {path}:{line_number}"
            )
        case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise EvaluationDatasetError(f"Dataset {path} does not contain evaluation cases")
    return tuple(cases)
