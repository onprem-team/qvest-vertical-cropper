"""FastAPI application for asynchronous vertical-crop jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from v_cropper import __version__

from .config import Settings
from .config import settings as default_settings
from .keys import (
    BOOTSTRAP_KEY_ID,
    ChainVerifier,
    KeyStore,
    Principal,
    StaticTokenVerifier,
)
from .models import CropJobRequest, JobAccepted, JobView, KeyCreated, KeyCreateRequest, KeyView
from .security import (
    MIN_RECOMMENDED_TOKEN_LENGTH,
    UnsafeURLError,
    check_public_origin,
    extract_bearer_token,
    validate_transport_url,
)
from .store import JobStore
from .worker import JobWorker

logger = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}


def create_app(settings: Settings | None = None, redis: Redis | None = None) -> FastAPI:
    config = settings or default_settings
    # Raises before anything binds: refusing to start is the whole point of the check.
    check_public_origin(config.public_origin, allow_plaintext=config.allow_plaintext)
    owns_redis = redis is None
    client = redis or Redis.from_url(config.redis_url)
    store = JobStore(client, ttl_sec=config.job_ttl_sec, event_ttl_sec=config.event_ttl_sec)
    keystore = KeyStore(client)
    verifier = ChainVerifier([StaticTokenVerifier(config.api_token), keystore])
    worker = JobWorker(store, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.redis = client
        app.state.store = store
        app.state.keystore = keystore
        app.state.worker = worker
        task = asyncio.create_task(worker.run(), name="v-cropper-worker")
        yield
        worker.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if owns_redis:
            await client.aclose()

    app = FastAPI(
        title="v-cropper service",
        description="Asynchronous VLM-guided 9:16 crop jobs using presigned object-storage URLs.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.redis = client
    app.state.store = store
    app.state.keystore = keystore
    app.state.worker = worker

    if config.api_token and len(config.api_token) < MIN_RECOMMENDED_TOKEN_LENGTH:
        logger.warning(
            "CROPPER_API_TOKEN is shorter than %d characters; it is the only thing standing "
            "between the internet and this service.",
            MIN_RECOMMENDED_TOKEN_LENGTH,
        )

    async def require_principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
        """Authenticate a caller and attach the principal to the request.

        The configured bootstrap token is always admin. Minted keys are ordinary
        callers. Fails closed when nothing is configured and nothing matches.
        """
        presented = extract_bearer_token(authorization)
        if presented:
            principal = await verifier.verify(presented)
            if principal is not None:
                request.state.principal = principal
                return principal
        if not config.api_token:
            raise HTTPException(status_code=503, detail="API token is not configured")
        raise HTTPException(
            status_code=401,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def require_admin(request: Request) -> Principal:
        principal = getattr(request.state, "principal", None)
        if principal is None or not principal.is_admin:
            raise HTTPException(status_code=403, detail="admin credential required")
        return principal

    # Job routes live on a router that carries the auth dependency, so a route added later
    # cannot end up unauthenticated by omission. Admin routes add a second check so a
    # minted key can never mint or revoke.
    v1 = APIRouter(prefix="/v1", tags=["jobs"], dependencies=[Depends(require_principal)])
    admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

    @app.get("/healthz", tags=["operations"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"])
    async def readyz() -> dict[str, str]:
        try:
            await client.ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Redis unavailable") from exc
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise HTTPException(status_code=503, detail="ffmpeg and ffprobe are required")
        provider = os.getenv("VCROPPER_PROVIDER", "openai").strip().lower()
        if provider == "bedrock":
            if not (os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION")):
                raise HTTPException(status_code=503, detail="AWS Bedrock region is not configured")
            # A Claude inference-profile fallback is used when no model env is set.
        elif provider == "openai":
            if not (os.getenv("VCROPPER_API_KEY") or os.getenv("GEMINI_API_KEY")):
                raise HTTPException(status_code=503, detail="VLM API key is not configured")
        else:
            raise HTTPException(status_code=503, detail="VCROPPER_PROVIDER must be openai or bedrock")
        if not config.api_token:
            raise HTTPException(status_code=503, detail="API token is not configured")
        return {"status": "ready"}

    @app.get("/version", tags=["operations"])
    async def version() -> dict[str, str]:
        return {"version": __version__}

    @v1.post("/jobs", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(payload: CropJobRequest) -> JobAccepted:
        try:
            for url in (payload.source_url, payload.destination_url):
                validate_transport_url(
                    url,
                    allowed_hosts=config.allowed_hosts,
                    allow_private=config.allow_private_hosts,
                )
        except UnsafeURLError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        duration = payload.out_s - (payload.in_s or 0) if payload.out_s is not None else None
        if duration is not None and duration > config.max_duration_sec:
            raise HTTPException(status_code=422, detail="requested clip exceeds duration limit")
        job_id = uuid.uuid4().hex
        await store.create(job_id, payload)
        return JobAccepted(job_id=job_id)

    @v1.get("/jobs/{job_id}", response_model=JobView)
    async def get_job(job_id: str) -> JobView:
        state = await store.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobView.model_validate(state)

    @v1.delete("/jobs/{job_id}", response_model=JobView, status_code=status.HTTP_202_ACCEPTED)
    async def cancel_job(job_id: str) -> JobView:
        if not await store.cancel(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        state = await store.get(job_id)
        return JobView.model_validate(state)

    @v1.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        if await store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def stream():
            cursor = 0
            while True:
                if await request.is_disconnected():
                    return
                events = await store.events(job_id, cursor)
                for event in events:
                    cursor += 1
                    event_name = "terminal" if event["status"] in TERMINAL else "progress"
                    public = JobView.model_validate(event).model_dump(mode="json")
                    yield f"event: {event_name}\ndata: {json.dumps(public)}\n\n"
                    if event["status"] in TERMINAL:
                        return
                yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @admin.post("/keys", response_model=KeyCreated, status_code=status.HTTP_201_CREATED)
    async def mint_key(payload: KeyCreateRequest) -> KeyCreated:
        try:
            token, record = await keystore.mint(payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return KeyCreated(token=token, **record)

    @admin.get("/keys", response_model=list[KeyView])
    async def list_keys() -> list[KeyView]:
        return [KeyView.model_validate(item) for item in await keystore.list()]

    @admin.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def revoke_key(key_id: str) -> None:
        if key_id == BOOTSTRAP_KEY_ID:
            raise HTTPException(
                status_code=400,
                detail="the bootstrap token is configuration and cannot be revoked through the API",
            )
        if not await keystore.revoke(key_id):
            raise HTTPException(status_code=404, detail="key not found")

    v1.include_router(admin)
    app.include_router(v1)
    return app


app = create_app()
