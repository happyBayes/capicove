"""Typed failures shared by pure video range planners."""

from capicove.processing.video.models import VideoSplitWarningCode


class PlanningFailure(ValueError):
    """Report invalid source metadata or strategy planning input."""

    def __init__(
        self,
        message: str,
        warning_code: VideoSplitWarningCode = VideoSplitWarningCode.INVALID_TIMESTAMPS,
    ) -> None:
        super().__init__(message)
        self.warning_code = warning_code
