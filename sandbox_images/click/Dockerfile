# Sandbox image for pallets/click (D2; docs/decisions/0003 and 0004).
#
# Dependencies: pallets/click's runtime dependencies plus its `tests` dependency group at
# image_commit 08b803cf32ceb2ae6a772c62d5da42757a897d44,
# frozen in requirements.lock by bootstrap.Dockerfile
# (`pip install . --group tests` from the source archive, then `pip uninstall -y click`).
# This file installs exactly that lock with --no-deps, so rebuilds are reproducible. The package
# itself is not installed: tests import the copy in /work through PYTHONPATH. Two click-specific
# additions follow: the `less` pager and a metadata-only dist-info for click.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
# `less` (Debian trixie, pinned): tests/test_utils/test_echo_via_pager.py runs the real pager.
RUN apt-get update \
 && apt-get install -y --no-install-recommends less=668-1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock
# Metadata only, no code: tests/test_deprecations.py calls importlib.metadata.version("click") at
# import time. The version is the one in pyproject.toml at image_commit (8.6.0.dev, normalised).
# Tests still import the code in /work/src (docs/decisions/0004-dataset-source-repos.md).
RUN d=/usr/local/lib/python3.12/site-packages/click-8.6.0.dev0.dist-info \
 && mkdir "$d" \
 && printf 'Metadata-Version: 2.1\nName: click\nVersion: 8.6.0.dev0\n' > "$d/METADATA" \
 && printf 'click-8.6.0.dev0.dist-info/METADATA,,\nclick-8.6.0.dev0.dist-info/RECORD,,\n' > "$d/RECORD"
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -M -d /tmp -s /usr/sbin/nologin sandbox
# Image contract (docs/decisions/0003-sandbox-copy-into-volume.md):
RUN mkdir /work && chown 1000:1000 /work
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp PYTHONPATH=/work/src
USER 1000:1000
WORKDIR /work
