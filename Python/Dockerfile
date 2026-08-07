# Container build for the C-OWPF Streamlit app.
# Railway auto-detects this Dockerfile and uses it INSTEAD of Railpack/Nixpacks,
# so deploys no longer depend on the railpack-frontend builder image.
# Build context / root directory is Python/ (same place as the Procfile).
FROM python:3.11.9-slim

# libgomp1: OpenMP runtime needed by the bundled EPANET (epyt) and some numeric wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the application (respects .dockerignore).
COPY . .

# Streamlit honors Railway's injected $PORT; default to 8501 for local `docker run`.
# Shell form so ${PORT} is expanded at runtime.
CMD streamlit run app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
