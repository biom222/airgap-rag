import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import UploadFile

from airgap_rag.documents.errors import DocumentTooLargeError, InvalidDocumentError


@dataclass(frozen=True, slots=True)
class StagedDocument:
    path: Path
    size: int
    sha256: str


class LocalDocumentStore:
    """Stores uploads under server-generated keys on a local filesystem."""

    def __init__(self, root: Path, max_upload_bytes: int, read_chunk_bytes: int) -> None:
        self._root = root.resolve()
        self._max_upload_bytes = max_upload_bytes
        self._read_chunk_bytes = read_chunk_bytes

    async def stage(self, upload: UploadFile) -> StagedDocument:
        staging_directory = self._root / ".staging"
        await to_thread.run_sync(staging_directory.mkdir, 0o750, True, True)
        path = staging_directory / f"{uuid4().hex}.upload"
        digest = hashlib.sha256()
        size = 0

        try:
            with path.open("xb") as destination:
                while chunk := await upload.read(self._read_chunk_bytes):
                    size += len(chunk)
                    if size > self._max_upload_bytes:
                        raise DocumentTooLargeError(self._max_upload_bytes)
                    digest.update(chunk)
                    await to_thread.run_sync(destination.write, chunk)
        except Exception:
            await self.discard(path)
            raise

        if size == 0:
            await self.discard(path)
            raise InvalidDocumentError("Пустой файл не может быть загружен.")

        return StagedDocument(path=path, size=size, sha256=digest.hexdigest())

    def make_storage_key(self, document_id: UUID, sha256: str, suffix: str) -> str:
        return f"documents/{document_id}/{sha256}{suffix}"

    async def promote(self, staged_path: Path, storage_key: str) -> Path:
        destination = self.resolve(storage_key)
        await to_thread.run_sync(destination.parent.mkdir, 0o750, True, True)
        await to_thread.run_sync(os.replace, staged_path, destination)
        return destination

    def resolve(self, storage_key: str) -> Path:
        destination = (self._root / storage_key).resolve()
        if not destination.is_relative_to(self._root):
            raise ValueError("storage key escapes document root")
        return destination

    async def discard(self, path: Path) -> None:
        await to_thread.run_sync(path.unlink, True)

    async def delete(self, storage_key: str) -> None:
        await self.discard(self.resolve(storage_key))
