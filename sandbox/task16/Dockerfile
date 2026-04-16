FROM python:3.12-slim
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY init_db.py /workspace/init_db.py
RUN python3 /workspace/init_db.py
