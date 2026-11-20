# Bootstrap that produced sandbox_images/click/requirements.lock (stage 0C). Not used for tests.
#
# Installs pallets/click's runtime dependencies and its `tests` dependency group as they were at
# image_commit 08b803cf32ceb2ae6a772c62d5da42757a897d44,
# from GitHub's source archive (the base image has no git), then uninstalls
# the package itself so tests import the copy in /work. pip 25.1 or later is needed for --group.
# Upstream build hooks run only here, inside docker build (D2). Regenerate the lock with:
#   docker build -f bootstrap.Dockerfile -t mapr-sandbox-click:bootstrap sandbox_images/click
#   docker run --rm mapr-sandbox-click:bootstrap pip freeze --all > requirements.lock
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG ARCHIVE=https://codeload.github.com/pallets/click/tar.gz/08b803cf32ceb2ae6a772c62d5da42757a897d44
RUN pip install --no-cache-dir pip==25.1.1 \
 && python -c "import io, sys, tarfile, urllib.request; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sys.argv[1]).read())).extractall('/tmp/src', filter='data')" "$ARCHIVE" \
 && cd /tmp/src/* \
 && pip install --no-cache-dir . --group tests \
 && pip uninstall -y click \
 && rm -rf /tmp/src
