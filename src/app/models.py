from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SentenceResponse(BaseModel):
    """Response model containing extracted sentences and processing metadata."""

    model_config = ConfigDict(populate_by_name=True)

    sentences: List[str] = Field(
        ..., description="List of segmented sentences extracted from the PDF body text."
    )
    method: Optional[str] = Field(
        None,
        description="Extraction engine used: 'fast_path' (digital text) or 'rapidocr_onnx' (vision OCR).",
    )
    page_count: Optional[int] = Field(
        None, description="Number of pages detected in the PDF document."
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Server-side processing duration in milliseconds."
    )


class InspectResponse(BaseModel):
    """Inspection result providing page count and estimated processing time."""

    model_config = ConfigDict(populate_by_name=True)

    page_count: int = Field(..., description="Total pages detected in the document.")
    has_text_layer: bool = Field(
        ..., description="True if native digital vector text was detected."
    )
    recommended_method: str = Field(
        ..., description="Recommended extraction path ('fast_path' or 'rapidocr_onnx')."
    )
    estimated_seconds: float = Field(
        ..., description="Estimated processing duration in seconds."
    )


class JobStatus(str, Enum):
    """Execution status for asynchronous background extraction tasks."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Asynchronous background job progress and result model."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    filename: Optional[str] = None
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_page: int = 0
    total_pages: int = 0
    message: Optional[str] = None
    result: Optional[SentenceResponse] = None
    created_at: float
    updated_at: float
    error: Optional[str] = None


class JobSubmitResponse(BaseModel):
    """Response returned upon submitting an asynchronous extraction task."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    status: JobStatus
    total_pages: int
    estimated_seconds: float

