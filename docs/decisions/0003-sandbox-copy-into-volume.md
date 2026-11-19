# 0003: Sandbox code is copied into an anonymous volume

Date: Nov 19, 2026 · Status: accepted · Refines D2 (does not replace it)

## Context

D2 says PR code is **copied** into a throwaway container with `put_archive`, never mounted from
the host, and that the container runs with `--network none --read-only --tmpfs /tmp --memory 1g
--cpus 1 --pids-limit 256`, a non-root user, `--rm` and a timeout. With `--read-only`, the root
filesystem cannot receive the copy, and `--rm` deletes the container, and its logs, as soon as
it exits.

## Decision

`src/reviewer/sandbox.py` (`run_in_sandbox`) does this:

1. **Copy into an anonymous volume.** The container is created (not started) with an anonymous
   Docker volume mounted at `/work` (`Mount(target="/work", source=None, type="volume")`).
   This is not a host mount: the volume lives inside Docker's own storage and has no host path.
   The code is copied in with `container.put_archive("/work", tar_bytes)` **before** the
   container starts. The tar is built in memory; every member is owned by uid/gid 1000, with
   mode 0o755 for directories and 0o644 for files. Docker accepts `put_archive` into a volume
   even though the root filesystem stays read-only (checked in stage 0B).
2. **Every other D2 setting is unchanged:** `network_mode="none"`, `read_only=True`,
   `tmpfs={"/tmp": "rw,size=256m"}`, `mem_limit="1g"`, `nano_cpus=1_000_000_000`,
   `pids_limit=256`, `working_dir="/work"`, and a timeout (`wait(timeout=...)`, then `kill()`).
3. **Non-root user.** The container runs as `user="1000:1000"`.
4. **Explicit removal replaces `--rm`.** After stdout and stderr have been read separately, a
   `finally` block calls `container.remove(v=True, force=True)`, which removes the container and
   its anonymous volume on every path: success, failure and timeout. Auto-removal would delete the
   logs before they could be read.

## Image contract: `/work` owned by 1000:1000

Root cause found in stage 0B: Docker's archive extraction skips the tar's `.` entry when the
destination directory already exists. Files and subdirectories in the tar get uid 1000, but the
root of the volume does not. A new anonymous volume instead takes its ownership from the image's
`/work` (Docker copies the image's directory into an empty volume). With stock
`python:3.12-slim`, which has no `/work`, the volume root is `root:root 0755` and uid 1000 gets
`EACCES` creating `/work/.write_test`.

So every sandbox image must contain:

```dockerfile
RUN mkdir /work && chown 1000:1000 /work
```

- `scripts/check_sandbox.py` builds `mapr-sandbox-base:check` from an in-memory Dockerfile:
  `FROM python:3.12-slim@sha256:<pulled digest>` plus exactly that line.
- Stage 2's per-repository Dockerfiles in `sandbox_images/` must include the same `RUN` line.
- There is no runtime ownership pre-check (it would cost an extra container); the stage 0B probe
  checks that `/work` is writable.

## Consequences

- D2's "copied, never mounted from the host" holds: there are no bind mounts of any kind.
- The read-only root filesystem is kept; writable places are `/tmp` (tmpfs) and `/work`.
- An image without `/work` owned by 1000:1000 still runs, but code cannot create files directly
  in `/work` (for example `.pytest_cache`).
