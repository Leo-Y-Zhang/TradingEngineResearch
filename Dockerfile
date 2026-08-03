# syntax=docker/dockerfile:1
# ── TradingEngineResearch — reproducible container (ROADMAP Phase 6, item 6) ──────
# Multi-stage: a builder installs the quant core + runtime extras into an isolated
# virtualenv against the pinned constraints.txt; the slim runtime image carries only
# that venv + the source, runs as a non-root user, and dispatches via entrypoint.sh.
#
#   docker build -t tradingengineresearch:latest .
#   docker run --rm -e ENGINE_MODE=RESEARCH -e ENGINE_UNIVERSE='["AAPL","MSFT"]' tradingengineresearch:latest loop
# Prefer docker-compose.yml for a real run (state volume, secrets mount, port map).

# ---- builder ---------------------------------------------------------------------
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build

# Packaging metadata first (better layer caching), then the source packages that
# [tool.setuptools.packages.find].include declares.
COPY pyproject.toml constraints.txt README.md alembic.ini ./
COPY core ./core
COPY data ./data
COPY research ./research
COPY strategies ./strategies
COPY nlp ./nlp
COPY execution ./execution
COPY learning ./learning
COPY ops ./ops
COPY broker ./broker
COPY backtesting ./backtesting
COPY monitoring ./monitoring
COPY migrations ./migrations

# Runtime extras: API + persistence + ingestion + vault. (Add ',brokers' for LIVE
# IBKR and ',nlp' for FinBERT — both heavy; omitted from the base image.)
RUN pip install --upgrade pip \
 && pip install ".[app,persistence,ingestion,vault]" -c constraints.txt

# ---- runtime ---------------------------------------------------------------------
FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENGINE_MODE=RESEARCH \
    ENGINE_PERSISTENCE__STATE_DIR=/app/state \
    ENGINE_VAULT__DIRECTORY=/app/secrets
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --from=builder /build /app
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && useradd --create-home --uid 10001 tradingengineresearch \
 && mkdir -p /app/state /app/secrets \
 && chown -R tradingengineresearch:tradingengineresearch /app
USER tradingengineresearch
EXPOSE 8000
# Container health: the API answers /health (only meaningful for the api/combined modes).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)" || exit 1
ENTRYPOINT ["entrypoint.sh"]
CMD ["combined"]
