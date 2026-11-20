# Bootstrap that produced sandbox_images/marshmallow/requirements.lock (stage 0C). Not used for tests.
#
# Installs marshmallow-code/marshmallow's runtime dependencies and its `tests` dependency group as they were at
# image_commit f268f0c4d12c4edb43743ca5703eed7954320245,
# from GitHub's source archive (the base image has no git), then uninstalls
# the package itself so tests import the copy in /work. pip 25.1 or later is needed for --group.
# Upstream build hooks run only here, inside docker build (D2). Regenerate the lock with:
#   docker build -f bootstrap.Dockerfile -t mapr-sandbox-marshmallow:bootstrap sandbox_images/marshmallow
#   docker run --rm mapr-sandbox-marshmallow:bootstrap pip freeze --all > requirements.lock
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG ARCHIVE=https://codeload.github.com/marshmallow-code/marshmallow/tar.gz/f268f0c4d12c4edb43743ca5703eed7954320245
RUN pip install --no-cache-dir pip==25.1.1 \
 && python -c "import io, sys, tarfile, urllib.request; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sys.argv[1]).read())).extractall('/tmp/src', filter='data')" "$ARCHIVE" \
 && cd /tmp/src/* \
 && pip install --no-cache-dir . --group tests \
 && pip uninstall -y marshmallow \
 && rm -rf /tmp/src
