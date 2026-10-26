# Globus Connect Personal (GCP) for the download of the Auriga snapshots.
#
# The Auriga release is served only through Globus, so a download needs a Globus
# collection on this machine. GCP is a daemon, not a Python package, so it cannot live in
# the uv .venv; it runs here as the globus service of compose.yaml, with write access to
# staging/auriga/ only. The transfers are submitted by `uv run experiments auriga
# download`; packages/experiments/README.md lists every step.

# GCP runs its control scripts with the python3 it finds on the PATH.
FROM python:3.14.5-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# GCP is published only as "latest"; the checksum pins version 3.3.2.766 and fails the
# build when Globus replaces the tarball.
ARG GCP_SHA256=a5c6251fcb795b92bec8ec95a321638154fdbe2ee061080c518a5aa2fc2dedae
RUN curl -fsSL -o /tmp/gcp.tgz \
        https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz \
    && echo "${GCP_SHA256}  /tmp/gcp.tgz" | sha256sum -c - \
    && tar -xzf /tmp/gcp.tgz -C /opt \
    && rm /tmp/gcp.tgz \
    && ln -s /opt/globusconnectpersonal-3.3.2/globusconnectpersonal /usr/local/bin/globusconnectpersonal

# GCP refuses to run as root. The config directory exists in the image so that the named
# volume mounted over it inherits the user's ownership.
RUN useradd --create-home --uid 1000 researcher \
    && mkdir -p /home/researcher/.globusonline /data/auriga \
    && chown researcher:researcher /home/researcher/.globusonline /data/auriga
USER researcher
WORKDIR /home/researcher
