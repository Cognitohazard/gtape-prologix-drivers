# TDS460A

Digital Oscilloscope - 4 channels, 400MHz, up to 15000 points.

## Import

```python
from gtape_prologix_drivers.instruments.tds460a import TDS460A, WaveformData
```

## Constructor

```python
TDS460A(adapter)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | PrologixAdapter | Connected adapter instance |

---

## Channel Detection

### get_active_channels

```python
scope.get_active_channels() -> list[str]
```

Detect which channels are currently displayed.

**Returns:** List of channel names (e.g., `["CH1", "CH2"]`)

---

## Record Length

### set_record_length

```python
scope.set_record_length(length: int) -> int
```

Set horizontal record length.

| Parameter | Type | Description |
|-----------|------|-------------|
| `length` | int | Number of points (500, 1000, 2500, 5000, 15000) |

**Returns:** Actual length set by scope

---

## Waveform Capture

### read_waveform

```python
scope.read_waveform(channel: str, record_length: int = None) -> WaveformData
```

Read waveform from channel.

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel` | str | Channel name ("CH1", "CH2", "CH3", "CH4") |
| `record_length` | int \| None | Override record length |

**Returns:** `WaveformData` object with `.time`, `.voltage` (numpy arrays), `.preamble` (dict)

---

## Error Handling

### check_errors

```python
scope.check_errors() -> str
```

Query scope for errors.

**Returns:** Error string or "0,No error" if none

---

## Complete Example

```python
from gtape_prologix_drivers import PrologixAdapter
from gtape_prologix_drivers.instruments.tds460a import TDS460A

adapter = PrologixAdapter(port="COM4", gpib_address=1)
scope = TDS460A(adapter)

# Get active channels
channels = scope.get_active_channels()
print(f"Active: {channels}")

# Set record length
scope.set_record_length(5000)

# Capture waveform
waveform = scope.read_waveform("CH1")
print(f"Points: {len(waveform.voltage)}")
print(f"Voltage range: {min(waveform.voltage):.3f} to {max(waveform.voltage):.3f} V")

adapter.close()
```

## Notes

- The TDS460A uses different GPIB commands than the TDS3000 series
- Maximum record length is 15000 points
- Binary transfer at 9600 baud takes ~30-60 seconds for 15000 points
