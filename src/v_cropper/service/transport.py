"""Presigned HTTP PUT output transport."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx


async def _file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    """Read a file without blocking the event loop or buffering it all in RAM."""
    stream = await asyncio.to_thread(path.open, "rb")
    try:
        while chunk := await asyncio.to_thread(stream.read, chunk_size):
            yield chunk
    finally:
        await asyncio.to_thread(stream.close)


async def upload_output(
    url: str,
    path: Path,
    *,
    timeout_sec: float = 300,
    max_bytes: int = 2_000_000_000,
) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"output exceeds {max_bytes} byte limit")
    headers = {"Content-Type": "video/mp4", "Content-Length": str(size)}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec)) as client:
        response = await client.put(url, content=_file_chunks(path), headers=headers)
        response.raise_for_status()
    return size
