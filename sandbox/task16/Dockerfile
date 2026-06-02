FROM python:3.12-slim
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash sandbox
WORKDIR /workspace
COPY sandbox/task16/init_db.py /workspace/init_db.py
RUN python3 /workspace/init_db.py && chown -R sandbox:sandbox /workspace
USER sandbox
