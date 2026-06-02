FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends bash coreutils grep gawk sed && rm -rf /var/lib/apt/lists/*
COPY data/server.log /workspace/server.log
WORKDIR /workspace
RUN useradd -m -s /bin/bash sandbox
RUN chown -R sandbox:sandbox /workspace
USER sandbox
