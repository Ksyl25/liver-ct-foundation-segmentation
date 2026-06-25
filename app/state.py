"""Small state helpers for the Streamlit MVP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadedCase:
    """Container for an uploaded image and mask pair."""

    image_name: str
    mask_name: str | None = None
