# Bootstrap that produced sandbox_images/alembic/requirements.lock (stage 0C). Not used for tests.
#
# Installs sqlalchemy/alembic's runtime dependencies and its `tests` dependency group as they were at
# image_commit 1bcba93ea0473c69b48e704f842ca96e9e5348f5,
# from GitHub's source archive (the base image has no git), then uninstalls
# the package itself so tests import the copy in /work. pip 25.1 or later is needed for --group.
# Upstream build hooks run only here, inside docker build (D2). Regenerate the lock with:
#   docker build -f bootstrap.Dockerfile -t mapr-sandbox-alembic:bootstrap sandbox_images/alembic
#   docker run --rm mapr-sandbox-alembic:bootstrap pip freeze --all > requirements.lock
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG ARCHIVE=https://codeload.github.com/sqlalchemy/alembic/tar.gz/1bcba93ea0473c69b48e704f842ca96e9e5348f5
RUN pip install --no-cache-dir pip==25.1.1 \
 && python -c "import io, sys, tarfile, urllib.request; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sys.argv[1]).read())).extractall('/tmp/src', filter='data')" "$ARCHIVE" \
 && cd /tmp/src/* \
 && pip install --no-cache-dir . --group tests \
 && pip uninstall -y alembic \
 && rm -rf /tmp/src
