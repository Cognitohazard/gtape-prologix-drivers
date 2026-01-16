"""GTAPE Prologix Drivers - Lab instrument control via Prologix GPIB-USB.

Supported instruments:
- HP33120A: Arbitrary Waveform Generator
- TDS460A: Digital Oscilloscope
- AgilentE3631A: Triple Output Power Supply
- HP34401A: 6.5 Digit Multimeter
- PLZ164W: Electronic Load
"""

from .adapter import PrologixAdapter
from .instruments import (
    HP33120A,
    TDS460A,
    AgilentE3631A,
    HP34401A,
    PLZ164W,
)
from .instruments.tds460a import WaveformData

__version__ = "0.1.0"
__all__ = [
    "PrologixAdapter",
    "HP33120A",
    "TDS460A",
    "AgilentE3631A",
    "HP34401A",
    "PLZ164W",
    "WaveformData",
]
