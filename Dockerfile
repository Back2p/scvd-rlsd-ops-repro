ARG BASE_IMAGE=pytorch/pytorch@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
FROM ${BASE_IMAGE}
WORKDIR /workspace/ops-repro
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt
COPY . ./
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
# Container startup is read-only configuration inspection. It does not start workers or training.
CMD ["python", "-m", "ops_repro.doctor", "--config", "configs/runtime.example.json"]
