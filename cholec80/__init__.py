"""Modern PyTorch pipelines for the Cholec80 laparoscopic-surgery dataset.

Sub-packages
------------
cholec80.annotations   -- annotation parsing, index building, frame extraction
cholec80.data          -- torch Datasets (frame / sequence / clip / CycleGAN)
cholec80.models        -- EndoNet, SlowFast, CycleGAN, temporal heads
cholec80.metrics       -- phase / tool / video-level metrics (pure numpy)
cholec80.engine        -- training utilities (optimizer, AMP, checkpoints, logging)
cholec80.active_learning -- query strategies + labeled/unlabeled pool
"""

__version__ = "0.2.0"

from .constants import PHASES, TOOLS  # noqa: F401

__all__ = ["PHASES", "TOOLS", "__version__"]
