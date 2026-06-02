FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils git python3 sqlite3 xxd jq \
    && rm -rf /var/lib/apt/lists/*
COPY data/agentic/ /workspace/
RUN chmod +x /workspace/decode/cipher.sh
RUN cd /workspace && if [ -f git_forensics/setup.sh ]; then bash git_forensics/setup.sh; fi
WORKDIR /workspace
RUN useradd -m -s /bin/bash sandbox
RUN chown -R sandbox:sandbox /workspace
USER sandbox
