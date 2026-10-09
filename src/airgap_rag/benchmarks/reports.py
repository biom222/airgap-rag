from pathlib import Path

from pydantic import BaseModel


def write_json_report(report: BaseModel, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output
