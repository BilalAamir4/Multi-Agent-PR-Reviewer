# Sandbox image for marshmallow-code/marshmallow (D2; docs/decisions/0003 and 0004).
#
# Dependencies: marshmallow-code/marshmallow's runtime dependencies plus its `tests` dependency
# group at image_commit f268f0c4d12c4edb43743ca5703eed7954320245,
# frozen in requirements.lock by bootstrap.Dockerfile
# (`pip install . --group tests` from the source archive, then `pip uninstall -y marshmallow`).
# This file installs exactly that lock with --no-deps, so rebuilds are reproducible. The package
# itself is not installed: tests import the copy in /work through PYTHONPATH. One addition
# follows: a metadata-only dist-info for marshmallow.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock
# Metadata only, no code: tests/test_version_attributes.py (in commits up to 2025-01-04) reads
# importlib.metadata.version("marshmallow"). Version from pyproject.toml at image_commit.
# Tests still import the code in /work/src (docs/decisions/0004-dataset-source-repos.md).
RUN d=/usr/local/lib/python3.12/site-packages/marshmallow-4.4.0.dist-info \
 && mkdir "$d" \
 && printf 'Metadata-Version: 2.1\nName: marshmallow\nVersion: 4.4.0\n' > "$d/METADATA" \
 && printf 'marshmallow-4.4.0.dist-info/METADATA,,\nmarshmallow-4.4.0.dist-info/RECORD,,\n' \
    > "$d/RECORD"
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -M -d /tmp -s /usr/sbin/nologin sandbox
# Image contract (docs/decisions/0003-sandbox-copy-into-volume.md):
RUN mkdir /work && chown 1000:1000 /work
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp PYTHONPATH=/work/src
USER 1000:1000
WORKDIR /work
