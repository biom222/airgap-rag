from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    department: str | None = None
    document_type: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    tags: tuple[str, ...] = ()
    max_access_level: int | None = None
