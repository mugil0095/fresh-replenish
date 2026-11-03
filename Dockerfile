# The reorder API only: models run in the nightly batch, so the image has no ML stack.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --group serve
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# The batch's orders are mounted at /app/data/processed (see docker-compose.yml).
CMD ["uvicorn", "fresh_replenish.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
