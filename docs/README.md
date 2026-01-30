# Documentation

Welcome to the GTAPE Prologix Drivers documentation.

## Quick Links

- [Getting Started](getting-started.md) — Installation and basic usage
- [API Reference](api/) — Detailed API documentation for all instruments

## API Reference

| Module | Description |
|--------|-------------|
| [PrologixAdapter](api/adapter.md) | Core GPIB communication with auto-reconnect |
| [TDS3000 Series](api/tds3000.md) | TDS3054 (4ch, 500MHz), TDS3012B (2ch, 100MHz) oscilloscopes |
| [TDS460A](api/tds460a.md) | TDS460A 4-channel 400MHz oscilloscope |
| [HP34401A](api/hp34401a.md) | 6.5 digit multimeter (DC/AC voltage, current, resistance) |
| [AgilentE3631A](api/e3631a.md) | Triple output power supply (6V/5A, ±25V/1A) |
| [PLZ164W](api/plz164w.md) | Kikusui electronic load (165W, CC/CV/CR/CP modes) |
| [HP33120A](api/hp33120a.md) | 15MHz arbitrary waveform generator |

## Features

- **No NI-VISA required** — Pure pyserial implementation
- **Auto-reconnect** — Handles transient USB disconnects gracefully
- **Binary waveform transfer** — IEEE 488.2 block format support
- **Multi-instrument control** — Single adapter can switch between instruments
