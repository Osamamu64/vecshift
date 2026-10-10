# syntax=docker/dockerfile:1
# The vecshift CLI in a small image that runs as a non-root user.
#
#   docker build -t vecshift .
#   docker run --rm -e VECSHIFT_DSN vecshift doctor --table documents
#
# The base image is pinned by digest; Dependabot keeps it current. PYTHON_IMAGE can point
# at a mirror with the same digest, such as mirror.gcr.io/library/python.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

FROM ${PYTHON_IMAGE} AS build
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
# Dependencies come from the lock file, checked against its hashes.
RUN uv export --frozen --no-dev --no-emit-project -o /dist/requirements.txt \
    && uv build --wheel -o /dist

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="vecshift" \
      org.opencontainers.image.description="Zero-downtime embedding model migrations for pgvector and Supabase" \
      org.opencontainers.image.source="https://github.com/Osamamu64/vecshift" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --no-cache-dir --require-hashes -r /dist/requirements.txt \
    && pip install --no-cache-dir --no-deps /dist/*.whl \
    && useradd --create-home --uid 1000 --user-group vecshift
USER vecshift
WORKDIR /home/vecshift
ENTRYPOINT ["vecshift"]
CMD ["--help"]
