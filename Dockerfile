FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements.txt .
RUN python -m pip install --upgrade pip --no-cache-dir --retries 20 --timeout 300 --progress-bar off \
    && python -m pip install --no-cache-dir --retries 20 --timeout 300 --progress-bar off -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN mkdir -p data output/scenes output/audio

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
