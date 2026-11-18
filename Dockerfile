FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.8.0
WORKDIR /opt/loopinfer
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
EXPOSE 8000
ENTRYPOINT ["loopinfer"]
CMD ["--help"]
