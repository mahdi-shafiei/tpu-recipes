FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install --yes --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    cmake \
    python3.10 \
    python3-pip \
  && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get install --yes google-cloud-cli \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir /gcs


RUN update-alternatives --install \
    /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN git clone https://github.com/AI-Hypercomputer/maxtext.git && \
git clone https://github.com/AI-Hypercomputer/JetStream.git

RUN cd maxtext/ && \
bash setup.sh

RUN cd /JetStream && \
pip install -e .

RUN cd /JetStream/benchmarks && \
pip install -r requirements.in

# Reset working directory to /workspace
WORKDIR /workspace

ENTRYPOINT [ "/bin/bash" ]