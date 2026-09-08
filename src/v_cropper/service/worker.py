"""One bounded in-process worker consuming the Redis job queue."""
from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from pathlib import Path

from ..pipeline import CropOptions, PipelineCancelled, probe_media, run_pipeline
from .config import Settings
from .models import CropJobRequest
from .security import UnsafeURLError, validate_redirect_chain
from .store import JobStore
from .transport import upload_output

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(self, store: JobStore, settings: Settings):
        self.store = store
        self.settings = settings
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        """Consume jobs serially; this service intentionally has exactly one worker."""
        while not self._stopping.is_set():
            job_id = await self.store.dequeue(timeout=1)
            if job_id:
                await self.run_job(job_id)

    async def run_job(self, job_id: str) -> None:
        state = await self.store.get(job_id)
        request = await self.store.get_request(job_id)
        if not state or not request or state["status"] == "cancelled":
            return
        requested_duration = (
            request.out_s - (request.in_s or 0) if request.out_s is not None else None
        )
        if requested_duration is not None and requested_duration > self.settings.max_duration_sec:
            await self.store.update(
                job_id, status="failed", phase="failed",
                error={"code": "duration_limit", "message": "requested clip exceeds duration limit"},
            )
            return

        root = Path(self.settings.work_root) / job_id
        root.mkdir(parents=True, exist_ok=True)
        output = root / "output.mp4"
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        async def watch_cancel() -> None:
            while not cancel_event.is_set():
                if await self.store.is_cancelled(job_id):
                    cancel_event.set()
                    return
                await asyncio.sleep(0.25)

        def progress(phase: str, pct: float, detail: dict | None = None) -> None:
            future = asyncio.run_coroutine_threadsafe(
                self.store.update(job_id, status="running", phase=phase, progress_pct=pct),
                loop,
            )
            future.result(timeout=10)

        await self.store.update(job_id, status="running", phase="starting", progress_pct=1)
        watcher = asyncio.create_task(watch_cancel())
        try:
            # Re-checked here rather than only at submit: ffmpeg follows redirects, so the
            # whole chain has to clear the policy before the URL reaches it.
            await validate_redirect_chain(
                request.source_url,
                allowed_hosts=self.settings.allowed_hosts,
                allow_private=self.settings.allow_private_hosts,
                timeout_sec=self.settings.http_timeout_sec,
            )
            if requested_duration is None:
                source_media = await asyncio.to_thread(probe_media, request.source_url)
                if source_media["duration_sec"] > self.settings.max_duration_sec:
                    raise ValueError("source clip exceeds duration limit; provide a bounded trim")
            options = _options(request)
            result = await asyncio.to_thread(
                run_pipeline,
                request.source_url,
                output,
                options=options,
                in_s=request.in_s,
                out_s=request.out_s,
                work_dir=root,
                progress=progress,
                cancelled=cancel_event.is_set,
            )
            if result.media["duration_sec"] > self.settings.max_duration_sec:
                raise ValueError("processed clip exceeds duration limit")
            if cancel_event.is_set():
                raise PipelineCancelled("crop job cancelled")
            await self.store.update(job_id, phase="uploading", progress_pct=95)
            size = await upload_output(
                request.destination_url,
                result.output_path,
                timeout_sec=self.settings.http_timeout_sec,
                max_bytes=self.settings.max_output_bytes,
            )
            media = {**result.media, "size_bytes": size}
            await self.store.update(
                job_id, status="succeeded", phase="complete", progress_pct=100,
                metrics=result.metrics, result=media, error=None,
            )
        except PipelineCancelled:
            await self.store.update(
                job_id, status="cancelled", phase="cancelled",
                error={"code": "cancelled", "message": "job was cancelled"},
            )
        except UnsafeURLError as exc:
            # The messages are fixed policy strings, so echoing one leaks nothing.
            logger.warning("Crop job %s rejected an unsafe source URL: %s", job_id, exc)
            await self.store.update(
                job_id, status="failed", phase="failed",
                error={"code": "unsafe_source_url", "message": str(exc)},
            )
        except Exception as exc:
            # Exception text/tracebacks can contain complete signed URLs.
            logger.error("Crop job %s failed (%s)", job_id, type(exc).__name__)
            await self.store.update(
                job_id, status="failed", phase="failed",
                error={"code": "processing_failed", "message": "crop processing failed"},
            )
        finally:
            cancel_event.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            shutil.rmtree(root, ignore_errors=True)


def _options(request: CropJobRequest) -> CropOptions:
    return CropOptions(
        sport=request.sport,
        prompt=request.prompt,
        sample_fps=request.sample_fps,
        sample_every=request.sample_every,
        send_width=request.send_width,
        spring_k=request.spring_k,
        concurrency=request.concurrency,
        model=request.model,
        scoreboard=request.scoreboard,
        scoreboard_sample_count=request.scoreboard_sample_count,
        scoreboard_model=request.scoreboard_model,
        scoreboard_position=request.scoreboard_position,
        scoreboard_height_ratio=request.scoreboard_height_ratio,
        scoreboard_opacity=request.scoreboard_opacity,
    )
