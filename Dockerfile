FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    FRFRUIT_WORKSPACE=/home/user/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev --extra cloud \
    && mkdir -p /home/user/workspace \
    && chown -R user:user /app /home/user

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/health', timeout=3)"

CMD ["sh", "-c", "uv run --no-sync frfruit app --no-window --host 0.0.0.0 --port ${PORT:-7860} --workspace ${FRFRUIT_WORKSPACE:-/home/user/workspace}"]
