FROM ubuntu:22.04
RUN apt-get update && apt-get install -y bash coreutils && rm -rf /var/lib/apt/lists/*
COPY data/agentic/ /workspace/
RUN chmod +x /workspace/decode/cipher.sh
WORKDIR /workspace
RUN useradd -m -s /bin/bash sandbox
USER sandbox
