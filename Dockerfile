FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
COPY agents ./agents
COPY libs ./libs
COPY dev ./dev

WORKDIR /app/agents/mabool/api

RUN uv sync --frozen

ENV VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/agents/mabool/api"

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "mabool.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

