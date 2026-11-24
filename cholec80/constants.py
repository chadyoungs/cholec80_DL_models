"""Dataset-wide constants (single source of truth for class definitions)."""

from __future__ import annotations

#: Seven surgical phases of a laparoscopic cholecystectomy, in annotation order.
PHASES: tuple[str, ...] = (
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
)

#: Seven instrument classes annotated in Cholec80 (multi-label, may be co-present).
TOOLS: tuple[str, ...] = (
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
)

PHASE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(PHASES)}
NUM_PHASES: int = len(PHASES)
NUM_TOOLS: int = len(TOOLS)

#: Official tool annotations are given at 1 fps; phase annotations at video fps (25).
TOOL_ANNOTATION_FPS: int = 1
VIDEO_FPS: int = 25

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
