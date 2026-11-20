# Sandbox image for simonw/sqlite-utils (D2; docs/decisions/0003 and 0004).
#
# Dependencies: simonw/sqlite-utils's runtime dependencies plus its `dev` dependency group at
# image_commit 9c66bafa52ae23d40499c74b0db3b635bcddb876,
# frozen in requirements.lock by bootstrap.Dockerfile
# (`pip install . --group dev` from the source archive, then `pip uninstall -y sqlite-utils`).
# This file installs exactly that lock with --no-deps, so rebuilds are reproducible. The package
# itself is not installed: tests import the copy in /work through PYTHONPATH.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -M -d /tmp -s /usr/sbin/nologin sandbox
# Image contract (docs/decisions/0003-sandbox-copy-into-volume.md):
RUN mkdir /work && chown 1000:1000 /work
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp PYTHONPATH=/work
USER 1000:1000
WORKDIR /work
