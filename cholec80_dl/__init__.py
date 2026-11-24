"""Modern PyTorch pipelines for the Cholec80 laparoscopic-surgery dataset.

Sub-packages
------------
cholec80_dl.annotations   -- annotation parsing, index building, frame extraction
cholec80_dl.data          -- torch Datasets (frame / sequence / clip / CycleGAN)
cholec80_dl.models        -- EndoNet, SlowFast, CycleGAN, temporal heads
cholec80_dl.metrics       -- phase / tool / video-level metrics (pure numpy)
cholec80_dl.engine        -- training utilities (optimizer, AMP, checkpoints, logging)
cholec80_dl.active_learning -- query strategies + labeled/unlabeled pool
"""

__version__ = "0.3.0"

from .constants import PHASES, TOOLS  # noqa: F401

__all__ = ["PHASES", "TOOLS", "__version__"]
