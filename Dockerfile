FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system airgap \
    && useradd --system --gid airgap --create-home airgap

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /data/documents \
    && chown -R airgap:airgap /data/documents

USER airgap

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "airgap_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
