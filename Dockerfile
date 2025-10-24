FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.12
# EDITO Jupyter-gpu uses the image below
#FROM inseefrlab/onyxia-jupyter-pytorch:py3.13.7-gpu

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends jq && \
    rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl && \
    echo "source <(kubectl completion bash)" >> /home/${NB_USER}/.bashrc

# Install MinIO Client
RUN wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc

# Copy and set permissions of service's init script
COPY edito-init.sh /opt/edito-init.sh

RUN chmod +x /opt/edito-init.sh

RUN echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/notebook

COPY env.yaml /tmp/env.yaml

RUN mamba shell init --shell bash

#update environment and set permissions
RUN mamba env update -f /tmp/env.yaml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN chown -R ${NB_USER}:users /home/${NB_USER}

# why does this not work? (permissions are set to root, even after this line)
#USER ${NB_UID}

# move to notebook base directory and fill it with content
WORKDIR "/home/${NB_USER}/work"

COPY CITATION.cff LICENSE README.md .
COPY examples ./examples
COPY logo ./logo
COPY src ./src
COPY test ./test

# this should not be necessary
RUN chown -R ${NB_USER}:users .
USER ${NB_UID}

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip", "0.0.0.0"]

