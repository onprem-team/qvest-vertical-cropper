"""Public API contracts. Signed URLs only exist in the private request model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CropJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str = Field(min_length=8, max_length=8192, repr=False)
    destination_url: str = Field(min_length=8, max_length=8192, repr=False)
    in_s: float | None = Field(default=None, ge=0)
    out_s: float | None = Field(default=None, gt=0)
    sport: str = "football"
    prompt: str | None = Field(default=None, min_length=1, max_length=10000)
    sample_fps: float = Field(default=2.0, gt=0, le=30)
    sample_every: int | None = Field(default=None, ge=1)
    send_width: int = Field(default=768, ge=64, le=4096)
    spring_k: float = Field(default=0.05, gt=0, le=100)
    concurrency: int = Field(default=8, ge=1, le=32)
    model: str | None = Field(default=None, max_length=512)
    scoreboard: bool = False
    scoreboard_sample_count: int = Field(default=10, ge=1, le=100)
    scoreboard_model: str | None = Field(default=None, max_length=512)
    scoreboard_position: Literal["bottom", "top"] = "bottom"
    scoreboard_height_ratio: float = Field(default=0.16, gt=0, lt=0.5)
    scoreboard_opacity: float = Field(default=0.85, ge=0, le=1)

    @model_validator(mode="after")
    def validate_trim(self) -> CropJobRequest:
        if self.out_s is not None and self.out_s <= (self.in_s or 0):
            raise ValueError("out_s must be greater than in_s")
        return self


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobError(BaseModel):
    code: str
    message: str


class JobResult(BaseModel):
    content_type: str = "video/mp4"
    size_bytes: int
    duration_sec: float
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    phase: str
    progress_pct: float = Field(ge=0, le=100)
    metrics: dict[str, Any] | None = None
    result: JobResult | None = None
    error: JobError | None = None


class KeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)


class KeyView(BaseModel):
    key_id: str
    name: str
    created_at: str
    last_used_at: str | None = None


class KeyCreated(KeyView):
    token: str = Field(repr=False)
