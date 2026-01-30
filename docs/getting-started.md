# Getting Started

## Installation

```bash
# From PyPI
pip install gtape-prologix-drivers

# With numpy support (required for oscilloscopes and AWG)
pip install gtape-prologix-drivers[all]

# For development
git clone https://github.com/Cognitohazard/gtape-prologix-drivers.git
cd gtape-prologix-drivers
pip install -e ".[dev,all]"
```

## Basic Usage

```python
from gtape_prologix_drivers import PrologixAdapter, HP34401A

# Create adapter - this handles all GPIB communication
adapter = PrologixAdapter(port="COM4", gpib_address=22)

# Create instrument instance
dmm = HP34401A(adapter)

# Take a measurement
voltage = dmm.measure_voltage(range_volts=10)
print(f"Voltage: {voltage:.6f} V")

# Always close when done
adapter.close()
```

## Multi-Instrument Setup

A single Prologix adapter can control multiple instruments by switching GPIB addresses:

```python
from gtape_prologix_drivers import (
    PrologixAdapter,
    AgilentE3631A,
    HP34401A,
    PLZ164W
)

adapter = PrologixAdapter(port="COM4", gpib_address=5)

# Create all instruments
psu = AgilentE3631A(adapter)   # GPIB 5
dmm = HP34401A(adapter)        # GPIB 22
load = PLZ164W(adapter)        # GPIB 10

# Use PSU (already at address 5)
psu.configure_output(AgilentE3631A.P6V, voltage=12.0, current_limit=2.0)
psu.enable_output(True)

# Switch to load
adapter.switch_address(10)
load.configure_cc_mode(current=0.5)
load.enable_input(True)

# Switch to DMM
adapter.switch_address(22)
voltage = dmm.measure_voltage(range_volts=100)

# Cleanup
adapter.switch_address(10)
load.enable_input(False)
adapter.switch_address(5)
psu.enable_output(False)
adapter.close()
```

## Oscilloscope Waveform Capture

```python
from gtape_prologix_drivers import PrologixAdapter
from gtape_prologix_drivers.instruments.tds3000 import TDS3012B

adapter = PrologixAdapter(port="COM4", gpib_address=21)
scope = TDS3012B(adapter)

# Check which channels are active
channels = scope.get_active_channels()
print(f"Active channels: {channels}")

# Read waveform data
waveform = scope.read_waveform("CH1")

# waveform contains:
# - waveform.time: numpy array of time values (seconds)
# - waveform.voltage: numpy array of voltage values (volts)
# - waveform.preamble: dict with metadata (sample rate, etc.)

print(f"Points: {len(waveform.voltage)}")
print(f"Time span: {waveform.time[0]:.6e} to {waveform.time[-1]:.6e} s")
print(f"Voltage range: {min(waveform.voltage):.3f} to {max(waveform.voltage):.3f} V")

adapter.close()
```

## Hardware Requirements

- **Prologix GPIB-USB Controller** - This library is specifically designed for Prologix adapters
- **Python 3.10+**
- **pyserial** - Installed automatically
- **numpy** - Required for oscilloscopes and AWG (install with `[awg]` extra)

## Finding Your COM Port

### Windows
1. Open Device Manager
2. Look under "Ports (COM & LPT)"
3. Find "USB Serial Port" or similar

### Linux
```bash
ls /dev/ttyUSB*
```

### macOS
```bash
ls /dev/tty.usbserial*
```

## Finding GPIB Addresses

Each instrument has a GPIB address (0-30) set via its front panel or rear switches. Common defaults:

| Instrument | Typical Address |
|------------|-----------------|
| HP34401A | 22 |
| E3631A | 5 |
| PLZ164W | 1 or 10 |
| TDS scopes | 1 |

Consult your instrument's manual to find or change its GPIB address.
