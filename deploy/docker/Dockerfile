# One image for both roles: `fedmed server ...` (coordinator) and `fedmed node ...` (hospital).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch keeps the image ~4x smaller than the CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && rm -rf /root/.cache

# Run as an unprivileged user; certificates and data are mounted read-only at runtime
RUN useradd --create-home --uid 10001 fedmed && mkdir -p /work && chown fedmed /work
USER fedmed
WORKDIR /work

ENTRYPOINT ["fedmed"]
CMD ["--help"]
