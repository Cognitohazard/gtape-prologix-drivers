<p align="center">
  <h1 align="center">GTAPE Prologix Drivers</h1>
  <p align="center">
    Python drivers for lab instruments via Prologix GPIB-USB
    <br />
    <strong>No NI-VISA required</strong>
    <br />
    <br />
    <a href="https://pypi.org/project/gtape-prologix-drivers/"><img src="https://img.shields.io/pypi/v/gtape-prologix-drivers?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/gtape-prologix-drivers/"><img src="https://img.shields.io/pypi/pyversions/gtape-prologix-drivers" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
</p>

---

## Supported Instruments

| Category | Instruments |
|----------|-------------|
| **Oscilloscopes** | TDS3054, TDS3012B, TDS460A |
| **Multimeters** | HP34401A |
| **Power Supplies** | Agilent E3631A |
| **Electronic Loads** | Kikusui PLZ164W |
| **Waveform Generators** | HP33120A |

## Installation

```bash
pip install gtape-prologix-drivers
```

For oscilloscopes and AWG (requires numpy):
```bash
pip install gtape-prologix-drivers[awg]
```

## Quick Example

```python
from gtape_prologix_drivers import PrologixAdapter, HP34401A

adapter = PrologixAdapter(port="COM4", gpib_address=22)
dmm = HP34401A(adapter)

voltage = dmm.measure_voltage()
print(f"{voltage:.6f} V")

adapter.close()
```

## Documentation

📖 **[Getting Started](docs/getting-started.md)** — Installation and basic usage

📚 **API Reference:**

| Module | Description |
|--------|-------------|
| [PrologixAdapter](docs/api/adapter.md) | Core GPIB communication |
| [TDS3000 Series](docs/api/tds3000.md) | TDS3054, TDS3012B oscilloscopes |
| [TDS460A](docs/api/tds460a.md) | TDS460A oscilloscope |
| [HP34401A](docs/api/hp34401a.md) | Digital multimeter |
| [AgilentE3631A](docs/api/e3631a.md) | Power supply |
| [PLZ164W](docs/api/plz164w.md) | Electronic load |
| [HP33120A](docs/api/hp33120a.md) | Waveform generator |

## Typical Operation Timing

Measured delays for common instrument operations via Prologix GPIB-USB:

| Instrument | Operation | Typical | Range |
|------------|-----------|---------|-------|
| **HP34401A** | Voltage measurement (fixed range) | 75 ms | 73–90 ms |
| **Agilent E3631A** | Current measurement | 236 ms | 204–253 ms |
| **Kikusui PLZ164W** | Current read | 50 ms | 42–108 ms |

> **Note**: HP34401A in autorange mode can take up to seconds for near-zero readings. Use fixed range for consistent timing.

## Requirements

- Python 3.10+
- [Prologix GPIB-USB Controller](http://prologix.biz/gpib-usb-controller.html)

## License

MIT
