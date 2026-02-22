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
        browser: Optional reference to the browser instance (e.g. DrissionPage/Playwright page)
                 if keep_browser_open=True in SolverConfig. Use this to continue automation
                 in the same solved session.
    """

    token: str
    cookies: list[dict[str, Any]]
    time_taken: float
    captcha_type: CaptchaType
    attempts: int
    browser: Any | None = None  # ← ADDED: optional browser reference


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


CLASS_NAMES: list[dict[str, list[str]]] = [
    {
        "a fire hydrant": [
            "a fire hydrant",
            "fire hydrants",
            "гидрантами",
            "пожарные гидранты",
            "消防栓",
            "bocas de incendios",
            "una boca de incendios",
            "borne d'incendie",
            "bouches d'incendie",
            "Hydranten",
            "Feuerhydranten",
            "een brandkraan",
            "brandkranen",
            "idrante",
            "idranti",
            "um hidrante",
            "hidrantes",
        ],
    },
    {
        "bicycles": [
            "bicycles",
            "велосипеды",
            "自行车",
            "bicicletas",
            "vélos",
            "Fahrrädern",
            "fietsen",
            "biciclette",
        ],
    },
    # ... (all other entries remain unchanged) ...
    {
        "traffic_lights": [
            "traffic lights",
            "светофоры",
            "红绿灯",
            "semáforos",
            "feux de circulation",
            "Ampeln",
            "verkeerslichten",
            "semafori",
        ],
    },
]


def _normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower()


def _build_multilang_mappings(
    class_names: list[dict[str, list[str]]],
    class_ids: dict[str, int],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for entry in class_names:
        for label, synonyms in entry.items():
            class_id = class_ids.get(label)
            if class_id is None:
                continue
            mapping[_normalize_keyword(label)] = class_id
            for synonym in synonyms:
                mapping[_normalize_keyword(synonym)] = class_id
    return mapping


_MULTI_LANGUAGE_TARGET_IDS: dict[str, int] = {
    "a fire hydrant": 6,
    "bicycles": 0,
    "bridges": 1,
    "buses": 2,
    "cars": 3,
    "chimneys": 4,
    "crosswalks": 5,
    "motorcycles": 7,
    "mountains or hills": 8,
    "palm trees": 10,
    "stairs": 11,
    "taxis": 3,
    "tractors": 12,
    "traffic_lights": 13,
}

_MULTI_LANGUAGE_COCO_IDS: dict[str, int] = {
    "a fire hydrant": 10,
    "bicycles": 1,
    "boats": 8,
    "buses": 5,
    "cars": 2,
    "motorcycles": 3,
    "parking meters": 12,
    "taxis": 2,
    "traffic_lights": 9,
}

# Target keyword to YOLO class index mapping for the classification model.
_BASE_TARGET_MAPPINGS: dict[str, int] = {
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

TARGET_MAPPINGS: dict[str, int] = {
    **_BASE_TARGET_MAPPINGS,
    **_build_multilang_mappings(CLASS_NAMES, _MULTI_LANGUAGE_TARGET_IDS),
}

# Class mappings for YOLO detection model (yolo12x.pt).
_BASE_COCO_TARGET_MAPPINGS: dict[str, int] = {
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

COCO_TARGET_MAPPINGS: dict[str, int] = {
    **_BASE_COCO_TARGET_MAPPINGS,
    **_build_multilang_mappings(CLASS_NAMES, _MULTI_LANGUAGE_COCO_IDS),
}
