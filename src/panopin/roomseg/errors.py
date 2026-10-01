"""The one exception the room segmenter raises."""


class SegmentationError(RuntimeError):
    """The cloud cannot be segmented as asked; the message says why and what to check."""
