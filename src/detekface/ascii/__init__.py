"""Terminal ASCII face renderer package."""

from detekface.ascii.renderer import (
    ASCII_RAMP_BLOCKS,
    ASCII_RAMP_DETAILED,
    ASCII_RAMP_SHORT,
    frame_to_ascii,
)

__all__ = [
    "frame_to_ascii",
    "ASCII_RAMP_SHORT",
    "ASCII_RAMP_BLOCKS",
    "ASCII_RAMP_DETAILED",
]
