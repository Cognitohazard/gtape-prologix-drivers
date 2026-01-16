# GTAPE Prologix Drivers

Python drivers for controlling lab instruments via the Prologix GPIB-USB controller. Pure pyserial implementation with no NI-VISA dependency.

## Supported Instruments

| Instrument | Description |
|------------|-------------|
| **HP33120A** | Arbitrary Waveform Generator (12-bit DAC, 8-16000 points) |
| **TDS460A** | Digital Oscilloscope (4 channels, 16-bit, up to 15000 points) |
| **Agilent E3631A** | Triple Output Power Supply (6V/5A, +/-25V/1A) |
| **HP34401A** | 6.5 Digit Multimeter (DC/AC V/I, 2/4-wire resistance) |
| **PLZ164W** | Kikusui Electronic Load (CC/CV/CR/CP modes, 165W) |

## Installation

```bash
# From PyPI (when published)
pip install gtape-prologix-drivers

# From GitHub
pip install git+https://github.com/yourusername/gtape-prologix-drivers.git

# For development
git clone https://github.com/yourusername/gtape-prologix-drivers.git
cd gtape-prologix-drivers
pip install -e ".[dev]"
```

## Quick Start

```python
from gtape_prologix_drivers import PrologixAdapter, HP34401A, AgilentE3631A

# Create adapter (handles all GPIB communication)
adapter = PrologixAdapter(port="COM4", gpib_address=22)

# Create instrument instances
dmm = HP34401A(adapter)
psu = AgilentE3631A(adapter)

# Use the DMM
voltage = dmm.measure_voltage(range_volts=10)
print(f"Voltage: {voltage:.6f} V")

# Switch to PSU (different GPIB address)
adapter.switch_address(3)
psu.configure_output(AgilentE3631A.P6V, voltage=5.0, current_limit=1.0)
psu.enable_output(True)

# Clean up
psu.enable_output(False)
adapter.close()
```

## Multi-Instrument Example

```python
from gtape_prologix_drivers import (
    PrologixAdapter,
    AgilentE3631A,
    HP34401A,
    PLZ164W
)

# Single adapter controls all instruments via GPIB address switching
adapter = PrologixAdapter(port="COM4", gpib_address=3)

psu = AgilentE3631A(adapter)  # GPIB 3
dmm = HP34401A(adapter)       # GPIB 22
load = PLZ164W(adapter)       # GPIB 10

# Configure PSU
psu.configure_output(AgilentE3631A.P6V, voltage=12.0, current_limit=2.0)
psu.enable_output(True)

# Configure load
adapter.switch_address(10)
load.configure_cc_mode(current=0.5)
load.enable_input(True)

# Measure with DMM
adapter.switch_address(22)
voltage = dmm.measure_voltage(range_volts=100)
print(f"Output voltage: {voltage:.3f} V")

# Read PSU current
adapter.switch_address(3)
current = psu.measure_current()
print(f"Input current: {current:.4f} A")

# Cleanup
adapter.switch_address(10)
load.enable_input(False)
adapter.switch_address(3)
psu.enable_output(False)
adapter.close()
```

## API Reference

### PrologixAdapter

```python
adapter = PrologixAdapter(port, gpib_address, timeout=6.0, max_retries=3)

adapter.write(command)           # Send SCPI command
adapter.read()                   # Read response
adapter.ask(command)             # Write + read (query)
adapter.switch_address(addr)     # Change GPIB address
adapter.write_binary(cmd, data)  # Send binary data (IEEE 488.2 format)
adapter.read_binary(expected)    # Read binary data
adapter.close()                  # Close connection
```

### HP34401A (Multimeter)

```python
dmm = HP34401A(adapter)

# Quick measurements (configure + read in one call)
voltage = dmm.measure_voltage(ac=False, range_volts=10)
current = dmm.measure_current(ac=False, range_amps=1)
resistance = dmm.measure_resistance(four_wire=False, range_ohms=1000)

# Configure once, read many times (faster for repeated measurements)
dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)
reading = dmm.read()  # Fast after configure
```

### AgilentE3631A (Power Supply)

```python
psu = AgilentE3631A(adapter)

# Configure and enable
psu.configure_output(AgilentE3631A.P6V, voltage=5.0, current_limit=1.0)
psu.enable_output(True)

# Or step-by-step
psu.select_channel(AgilentE3631A.P25V)
psu.set_voltage(12.0)
psu.set_current_limit(0.5)
psu.enable_output(True)

# Measurements
voltage = psu.measure_voltage()
current = psu.measure_current()
```

### PLZ164W (Electronic Load)

```python
load = PLZ164W(adapter)

# Constant Current mode
load.configure_cc_mode(current=2.0, current_range=PLZ164W.CURR_RANGE_HIGH)
load.enable_input(True)

# Other modes
load.configure_cv_mode(voltage=12.0)   # Constant Voltage
load.configure_cr_mode(resistance=50)  # Constant Resistance
load.configure_cp_mode(power=10.0)     # Constant Power

# Measurements
voltage = load.measure_voltage()
current = load.measure_current()
power = load.measure_power()

# Important: disable input before changing modes
load.enable_input(False)
```

### TDS460A (Oscilloscope)

```python
scope = TDS460A(adapter)

# Get active channels
channels = scope.get_active_channels()  # ['CH1', 'CH2', ...]

# Set record length
scope.set_record_length(5000)

# Read waveform
waveform = scope.read_waveform('CH1')
print(f"Time: {waveform.time}")      # numpy array
print(f"Voltage: {waveform.voltage}") # numpy array
print(f"Metadata: {waveform.preamble}")
```

### HP33120A (AWG)

```python
import numpy as np
from gtape_prologix_drivers import HP33120A

awg = HP33120A(adapter)

# Create waveform (0-2047 range, 8-16000 points)
waveform = np.linspace(0, 2047, 100, dtype=np.uint16)

# Upload and configure
awg.setup_arbitrary_waveform(
    waveform,
    name="MYWFM",
    frequency=1000,  # Hz
    voltage=1.0,     # Vpp
    load=50          # Ohms
)
```

## Hardware Tests

The `hardware_tests/` directory contains scripts for verifying driver functionality with actual instruments:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run hardware tests (adjust COM port and GPIB addresses)
python hardware_tests/test_e3631a_hardware.py COM4 --addr 3
python hardware_tests/test_hp34401a_hardware.py COM4 --addr 22
python hardware_tests/test_plz164w_hardware.py COM4 --addr 10
python hardware_tests/test_tds460a_hardware.py COM4 --addr 1
```

## Unit Tests

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.10+
- pyserial >= 3.5
- numpy >= 1.24 (optional, for AWG waveforms)
- Prologix GPIB-USB Controller

## License

MIT License - see [LICENSE](LICENSE) for details.
