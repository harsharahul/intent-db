# IntentDB - local-first, intent-aware vector database
#
# Build:    docker build -t intentdb .
# CLI:      docker run --rm -v "$PWD/data:/data" intentdb init /data/kb.intentdb
# MCP:      docker run --rm -i -v "$PWD/data:/data" intentdb serve-mcp /data/kb.intentdb
FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

LABEL org.opencontainers.image.title="IntentDB" \
      org.opencontainers.image.description="A local-first, intent-aware vector database for LLM retrieval" \
      org.opencontainers.image.source="https://github.com/harsharahul/intent-db" \
      org.opencontainers.image.licenses="MIT"

COPY --from=build /install /usr/local

# Databases live on a mounted volume; run as an unprivileged user.
RUN useradd --create-home --uid 1000 intentdb \
    && mkdir /data && chown intentdb /data
USER intentdb
WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["intentdb"]
CMD ["--help"]
