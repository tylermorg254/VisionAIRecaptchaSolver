"""Type definitions and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CaptchaType(Enum):
    """Supported captcha types."""

    DYNAMIC_3X3 = "dynamic_3x3"
    SELECTION_3X3 = "selection_3x3"
    SQUARE_4X4 = "square_4x4"
    INVISIBLE = "invisible"
    NO_CHALLENGE = "no_challenge"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Result of a successful captcha solve operation.

    Attributes:
        token: The reCAPTCHA response token.
        cookies: List of cookies from the browser session.
        time_taken: Time in seconds to solve the captcha.
        captcha_type: The type of captcha that was solved.
        attempts: Number of attempts made to solve.
    """

    token: str
    cookies: list[dict[str, Any]]
    time_taken: float
    captcha_type: CaptchaType
    attempts: int


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Result of YOLO object detection on a captcha image.

    Attributes:
        answers: List of grid cell indices containing the target object.
        confidence: Average confidence score of detections.
        target_class: The YOLO class index that was searched for.
    """

    answers: list[int]
    confidence: float
    target_class: int


# Target keyword to YOLO class index mapping for the classification model.
# Note: Class 9 ("other") is intentionally not mapped as it represents
# non target background images that should not match any reCAPTCHA target.
TARGET_MAPPINGS: dict[str, int] = {
    "bicycle": 0,
    "bicycles": 0,
    "bridge": 1,
    "bridges": 1,
    "bus": 2,
    "buses": 2,
    "car": 3,
    "cars": 3,
    "automobile": 3,
    "taxi": 3,
    "taxis": 3,
    "chimney": 4,
    "chimneys": 4,
    "crosswalk": 5,
    "crosswalks": 5,
    "hydrant": 6,
    "hydrants": 6,
    "fire hydrant": 6,
    "motorcycle": 7,
    "motorcycles": 7,
    "mountain": 8,
    "mountains": 8,
    "palm": 10,
    "palm tree": 10,
    "palm trees": 10,
    "stair": 11,
    "stairs": 11,
    "tractor": 12,
    "tractors": 12,
    "traffic": 13,
    "traffic light": 13,
    "traffic lights": 13,
}

# Class mappings for YOLO detection model (yolo12x.pt).
COCO_TARGET_MAPPINGS: dict[str, int] = {
    "bicycle": 1,
    "bicycles": 1,
    "car": 2,
    "cars": 2,
    "automobile": 2,
    "taxi": 2,
    "taxis": 2,
    "motorcycle": 3,
    "motorcycles": 3,
    "bus": 5,
    "buses": 5,
    "boat": 8,
    "boats": 8,
    "traffic": 9,
    "traffic light": 9,
    "traffic lights": 9,
    "hydrant": 10,
    "hydrants": 10,
    "fire hydrant": 10,
}
