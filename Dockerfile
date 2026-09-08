# syntax=docker/dockerfile:1.9
# A command-line image, not a service.
#
# `dslm` is a tool a CI job invokes: it reads files, computes, writes reports and
# exits. There is no listener, no port and no health endpoint. The HEALTHCHECK
# runs the tool's own self-check, which is the honest equivalent — it generates a
# corpus, curates, splits, tokenises, trains the control and scores it, and exits
# non-zero if the installed package cannot.
#
# Three things in here were learned the expensive way and are worth the comments:
# UV_PROJECT_ENVIRONMENT, --no-editable, and an ENTRYPOINT that names a
# subcommand.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.10.10

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# UV_PROJECT_ENVIRONMENT, not just VIRTUAL_ENV: uv resolves the project
# environment from the former. Without it `uv sync` installs into /build/.venv,
# and the image ships an empty /opt/venv that starts and then fails on the first
# import.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build

# Dependencies first, so the layer is reused whenever only source changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "${VIRTUAL_ENV}" \
 && uv sync --locked --no-default-groups --no-install-project

COPY src ./src
# --no-editable: uv installs a workspace project in editable mode by default,
# leaving a .pth that points at /build/src — a path the runtime stage does not
# have.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

RUN set -eux; \
    groupadd --system --gid 10001 app; \
    useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app; \
    install -d -o app -g app /app /app/work /app/reports

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /build/src /app/src
# The worked example ships in the image so the smoke test has a real corpus, a
# real split and a real trained model to gate — and so `docker run <image>
# evaluate ...` does something meaningful out of the box. It is about 1.2 MB,
# which is a cheap layer for a working demonstration.
COPY --chown=app:app examples /app/examples

WORKDIR /app
USER app

# Logs go to stderr; the report goes to stdout. JSON by default so a container
# log collector can parse it.
ENV DSLM_LOG__FORMAT=json \
    DSLM_LOG__LEVEL=INFO

# No port, so no HTTP health check. `doctor` runs a whole pipeline on a small
# generated corpus — generate, curate, split, tokenise, train, score — and exits
# non-zero if the installed package cannot. That is the strongest statement this
# image can make about itself without being given a file.
HEALTHCHECK --interval=60s --timeout=15s --start-period=5s --retries=2 \
  CMD ["dslm", "doctor"]

# ENTRYPOINT names the executable, CMD names the subcommand. An ENTRYPOINT that
# stops at the executable makes `docker run IMAGE` print usage and exit
# non-zero, which an orchestrator reads as a crash loop.
ENTRYPOINT ["dslm"]
CMD ["doctor"]
