# syntax=docker/dockerfile:1

# ---- build -----------------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip build \
 && python -m build --wheel --outdir /wheels

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MUHAQQIQ_CORPUS_DIR=/app/data/corpus \
    MUHAQQIQ_SKILLS_DIR=/app/skills \
    MUHAQQIQ_DB_PATH=/app/.muhaqqiq/runs.db \
    MUHAQQIQ_OUTPUT_DIR=/app/out

WORKDIR /app

COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels

# The corpus and the skills are data, not code: they are mounted next to the
# application so they can be edited or replaced without rebuilding the image.
COPY data ./data
COPY skills ./skills

RUN useradd --create-home --uid 10001 muhaqqiq \
 && mkdir -p /app/.muhaqqiq /app/out \
 && chown -R muhaqqiq:muhaqqiq /app
USER muhaqqiq

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "muhaqqiq.api:app", "--host", "0.0.0.0", "--port", "8000"]
