"""Workspace snapshots for the A2A transport (stage 3, decision 0011).

``pack`` makes a deterministic ``tar.gz`` of ``{posix_path: bytes}`` (sorted entries, fixed
mtime, owner and mode, gzip mtime 0): the same files always give the same bytes and so the same
SHA-256. ``unpack`` is the safe reader the service uses on bytes that came over the network. It
never uses ``extractall``; it reads the members one by one and refuses anything that is not a
plain file with a plain relative name: absolute paths, ``..``, backslashes, drive letters and
``:``, symlinks, hard links, devices, FIFOs, duplicate names, Windows device names, more than
``MAX_ARCHIVE_ENTRIES`` entries, more than ``MAX_UNPACKED_BYTES`` unpacked bytes, an archive over
``MAX_ARCHIVE_BYTES`` compressed. A bad archive raises ``ArchiveError`` before anything outside
the target directory can be touched.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import io
import tarfile
import zlib
from pathlib import Path, PurePosixPath

from reviewer.a2a_schema import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_ENTRIES, MAX_UNPACKED_BYTES

_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)
_CHUNK = 1024 * 1024
# Tar headers and padding on top of the file bytes (512-byte header per entry, padding, end blocks).
_TAR_OVERHEAD = MAX_ARCHIVE_ENTRIES * 1536 + 10240


class ArchiveError(ValueError):
    """The archive is not acceptable; the message says why."""


def check_member_name(name: str) -> str:
    """The name as a clean POSIX relative path, or ``ArchiveError``."""
    if not name or "\0" in name:
        raise ArchiveError("an entry has an empty name or a NUL in it")
    if "\\" in name:
        raise ArchiveError(f"entry {name!r}: backslashes are not allowed")
    if name.startswith("/") or ":" in name:
        raise ArchiveError(f"entry {name!r}: absolute paths and ':' are not allowed")
    parts = PurePosixPath(name).parts
    if not parts or "/".join(parts) != name or any(p in (".", "..") for p in parts):
        raise ArchiveError(f"entry {name!r}: not a plain relative path")
    for p in parts:
        stem = p.split(".", 1)[0].lower()
        if stem in _RESERVED or p != p.rstrip(" ."):
            raise ArchiveError(f"entry {name!r}: {p!r} is not a usable file name")
    return "/".join(parts)


def pack(files: dict[str, bytes]) -> bytes:
    """A deterministic tar.gz of ``files`` (regular files only, sorted by name)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(check_member_name(name))
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=6) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_sha256(data: bytes, expected: str) -> None:
    if not hmac.compare_digest(sha256_hex(data), expected):
        raise ArchiveError("workspace_sha256 does not match the archive")


class _CappedReader(io.RawIOBase):
    """Reads ``source`` and fails once more than ``limit`` bytes have come out of it (a gzip
    bomb stops here, not in memory)."""

    def __init__(self, source: io.BufferedIOBase, limit: int) -> None:
        self._source = source
        self._left = limit

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        data = self._source.read(len(buffer))
        self._left -= len(data)
        if self._left < 0:
            raise ArchiveError(f"the archive unpacks to more than {MAX_UNPACKED_BYTES} bytes")
        buffer[: len(data)] = data
        return len(data)


def unpack(data: bytes, dest: Path) -> int:
    """Unpack the tar.gz ``data`` into the existing directory ``dest`` and return the number of
    files written. Raises ``ArchiveError`` for any violation of the rules in the module
    docstring; files written before the error stay in ``dest`` (the caller deletes it)."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ArchiveError(f"the archive is larger than {MAX_ARCHIVE_BYTES} bytes")
    root = dest.resolve()
    seen: set[str] = set()
    total = 0
    try:
        gz = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
        capped = io.BufferedReader(_CappedReader(gz, MAX_UNPACKED_BYTES + _TAR_OVERHEAD))
        with tarfile.open(fileobj=capped, mode="r|") as tar:
            count = 0
            for member in tar:
                count += 1
                if count > MAX_ARCHIVE_ENTRIES:
                    raise ArchiveError(f"the archive has more than {MAX_ARCHIVE_ENTRIES} entries")
                if member.isdir():
                    check_member_name(member.name.rstrip("/"))
                    continue
                if not member.isreg():
                    kind = (
                        "a symbolic link" if member.issym()
                        else "a hard link" if member.islnk()
                        else "a device, FIFO or other special file"
                    )  # fmt: skip
                    raise ArchiveError(f"entry {member.name!r} is {kind}")
                name = check_member_name(member.name)
                if name in seen:
                    raise ArchiveError(f"entry {name!r} appears twice")
                seen.add(name)
                if member.size < 0 or total + member.size > MAX_UNPACKED_BYTES:
                    raise ArchiveError(
                        f"the archive unpacks to more than {MAX_UNPACKED_BYTES} bytes"
                    )
                total += member.size
                target = (root / PurePosixPath(name)).resolve()
                if not target.is_relative_to(root) or target == root:
                    raise ArchiveError(f"entry {name!r} would land outside the target directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    raise ArchiveError(f"entry {name!r} cannot be read")
                with target.open("wb") as out:
                    left = member.size
                    while left > 0:
                        chunk = src.read(min(_CHUNK, left))
                        if not chunk:
                            raise ArchiveError(f"entry {name!r} is truncated")
                        out.write(chunk)
                        left -= len(chunk)
    except ArchiveError:
        raise
    except (tarfile.TarError, OSError, EOFError, ValueError, zlib.error) as e:
        raise ArchiveError(f"the archive cannot be read: {type(e).__name__}: {e}") from e
    return len(seen)
