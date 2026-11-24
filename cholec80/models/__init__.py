"""Model zoo: EndoNet (frame + temporal), SlowFast, CycleGAN."""

from .backbone import Backbone, build_backbone
from .endonet import EndoNet, EndoNetOutput, TemporalEndoNet
from .slowfast import SlowFast, SlowFastOutput, build_slowfast
from .cyclegan import (
    Discriminator,
    Generator,
    ReplayBuffer,
    linear_decay_lambda,
    weights_init_normal,
)

__all__ = [
    "Backbone",
    "build_backbone",
    "EndoNet",
    "EndoNetOutput",
    "TemporalEndoNet",
    "SlowFast",
    "SlowFastOutput",
    "build_slowfast",
    "Generator",
    "Discriminator",
    "ReplayBuffer",
    "linear_decay_lambda",
    "weights_init_normal",
]
