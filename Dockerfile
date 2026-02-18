# ------------------------------------------------------------
# Stage 1: builder (micromamba)
# ------------------------------------------------------------
FROM mambaorg/micromamba:1.5.5 AS builder

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:$PATH

# tensorrt (pulled by tensorflow[and-cuda]) calls `ps` during build;
# `ps` is provided by the procps package (missing in micromamba base image)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /build
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /build/environment.yml

# Install into base instead of creating a new env
RUN micromamba install -y -n base -f /build/environment.yml && \
    micromamba clean --all --yes


# ------------------------------------------------------------
# Stage 2: runtime image
# ------------------------------------------------------------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# You use curl + unzip below for optional model download; install them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Copy conda env from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

# LD_LIBRARY_PATH expansion
ENV LD_LIBRARY_PATH=/opt/conda/lib

WORKDIR /app
COPY . .

# Optional model download (expects MODEL_ZIP_URL at build time)
ARG MODEL_ZIP_URL
RUN if [ -n "$MODEL_ZIP_URL" ]; then \
        echo "Downloading model bundle from $MODEL_ZIP_URL"; \
        mkdir -p models && \
        curl -L "$MODEL_ZIP_URL" -o /tmp/models.zip && \
        unzip /tmp/models.zip -d models && \
        rm /tmp/models.zip; \
    else \
        echo "MODEL_ZIP_URL not provided; skipping model download."; \
    fi

RUN rm -rf *.egg-info && pip install --no-cache-dir -e .

CMD ["tranquillyzer", "--help"]
