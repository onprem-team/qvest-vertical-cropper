from __future__ import annotations

import httpx
import pytest

from v_cropper.service import transport


@pytest.mark.asyncio
async def test_upload_output_streams_with_async_client(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"video-bytes")
    received = b""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal received
        received = await request.aread()
        assert request.headers["content-type"] == "video/mp4"
        assert request.headers["content-length"] == str(output.stat().st_size)
        return httpx.Response(200)

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(transport.httpx, "AsyncClient", client)

    assert await transport.upload_output("https://objects.example.test/output.mp4", output) == 11
    assert received == b"video-bytes"


@pytest.mark.asyncio
async def test_oversized_output_is_refused_before_any_request(tmp_path, monkeypatch):
    """The size cap has to be enforced locally, not by the object store rejecting a 2 GB body."""
    output = tmp_path / "output.mp4"
    output.write_bytes(b"x" * 128)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made for an oversized output")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        transport.httpx, "AsyncClient",
        lambda *a, **k: real_client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="exceeds 64 byte limit"):
        await transport.upload_output(
            "https://objects.example.test/output.mp4", output, max_bytes=64
        )


@pytest.mark.asyncio
async def test_upload_failure_surfaces_as_an_http_error(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"video-bytes")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        transport.httpx, "AsyncClient",
        lambda *a, **k: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(403, text="expired"))
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await transport.upload_output("https://objects.example.test/output.mp4", output)


@pytest.mark.asyncio
async def test_large_output_is_streamed_in_chunks(tmp_path, monkeypatch):
    """A multi-hundred-MB render must not be buffered into memory to upload it."""
    output = tmp_path / "output.mp4"
    output.write_bytes(b"ab" * 1024)
    chunks: list[int] = []

    original = transport._file_chunks

    def counting_chunks(path, chunk_size=1024 * 1024):
        async def wrapper():
            async for chunk in original(path, chunk_size=64):
                chunks.append(len(chunk))
                yield chunk

        return wrapper()

    monkeypatch.setattr(transport, "_file_chunks", counting_chunks)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        transport.httpx, "AsyncClient",
        lambda *a, **k: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ),
    )

    assert await transport.upload_output("https://objects.example.test/o.mp4", output) == 2048
    assert len(chunks) == 32
    assert max(chunks) == 64
