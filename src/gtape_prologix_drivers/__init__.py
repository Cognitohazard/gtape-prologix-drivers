"""GTAPE Prologix Drivers - Lab instrument control via Prologix GPIB-USB.

Supported instruments:
- HP33120A: Arbitrary Waveform Generator
- TDS460A: Digital Oscilloscope (TDS400 series)
- TDS3054: Digital Phosphor Oscilloscope (4ch, 500MHz)
- TDS3012B: Digital Phosphor Oscilloscope (2ch, 100MHz)
- AgilentE3631A: Triple Output Power Supply
- HP34401A: 6.5 Digit Multimeter
- PLZ164W: Electronic Load
"""

from .adapter import PrologixAdapter
from .instruments import (
    HP33120A,
    TDS460A,
    TDS3054,
    TDS3012B,
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
    "TDS3054",
    "TDS3012B",
    "AgilentE3631A",
    "HP34401A",
    "PLZ164W",
    "WaveformData",
]
