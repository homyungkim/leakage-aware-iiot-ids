# PALT edge-profile benchmark image (x86-64 host; emulates resource budgets, not ARM hardware)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir onnxruntime==1.25.0 numpy==2.4.4 lightgbm==4.7.0 xgboost-cpu==3.2.0
COPY bench.py /app/bench.py
WORKDIR /app
ENTRYPOINT ["python", "/app/bench.py"]
