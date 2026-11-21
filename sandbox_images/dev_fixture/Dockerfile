# Sandbox image for the synthetic dev fixtures in tests/fixtures/ (stage 2A, decision 0008).
#
# Not a dataset repository: eval/repos.yaml lists it as the local image `dev_fixture`
# (url: null, license: n/a), so `reviewer review --sandbox-repo dev_fixture` and the run_tests
# integration test can use it. The fixtures need only pytest; requirements.lock pins pytest
# 9.2.0 and its dependencies at the versions in this project's uv.lock and in the dataset
# images. Same base image, user and /work contract as the dataset images (decisions 0003, 0004).
#
#   docker build -t mapr-sandbox-dev_fixture:local sandbox_images/dev_fixture
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -M -d /tmp -s /usr/sbin/nologin sandbox
# Image contract (docs/decisions/0003-sandbox-copy-into-volume.md):
RUN mkdir /work && chown 1000:1000 /work
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp PYTHONPATH=/work
USER 1000:1000
WORKDIR /work
