# TDS460A

Digital Oscilloscope - 4 channels, 400MHz, 500-15000 points (up to 120k with Option 05).

## Import

```python
from gtape_prologix_drivers import TDS460A, WaveformData
```

## Constructor

```python
TDS460A(adapter)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | PrologixAdapter | Connected adapter instance |

---

## Identification & Setup

### get_id

```python
scope.get_id() -> str
```

Query instrument identification string.

**Returns:** IDN string (e.g., "TEKTRONIX,TDS 460A,...")

---

### autoset

```python
scope.autoset() -> None
```

Perform automatic setup (adjusts scales, triggers, etc.).

---

## Channel Detection & Display

### get_active_channels

```python
scope.get_active_channels() -> list[str]
```

Detect which channels are currently displayed.

**Returns:** List of channel names (e.g., `["CH1", "CH2"]`)

---

### set_channel_display

```python
scope.set_channel_display(channel: str, on: bool) -> None
```

Turn channel display on or off.

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel` | str | Channel name ("CH1"-"CH4") |
| `on` | bool | True to display, False to hide |

---

## Vertical (Channel) Settings

### get_channel_scale / set_channel_scale

```python
scope.get_channel_scale(channel: str) -> float
scope.set_channel_scale(channel: str, scale: float) -> None
```

Query or set volts/division for a channel.

---

### get_channel_offset / set_channel_offset

```python
scope.get_channel_offset(channel: str) -> float
scope.set_channel_offset(channel: str, offset: float) -> None
```

Query or set vertical offset in volts.

---

### get_channel_coupling / set_channel_coupling

```python
scope.get_channel_coupling(channel: str) -> str
scope.set_channel_coupling(channel: str, coupling: str) -> None
```

Query or set input coupling: "DC", "AC", or "GND".

---

### get_channel_bandwidth / set_channel_bandwidth

```python
scope.get_channel_bandwidth(channel: str) -> str
scope.set_channel_bandwidth(channel: str, bandwidth: str) -> None
```

Query or set bandwidth limit: "FULl", "TWEnty" (20MHz), or "ONEhundred" (100MHz).

---

## Horizontal Settings

### get_horizontal_scale / set_horizontal_scale

```python
scope.get_horizontal_scale() -> float
scope.set_horizontal_scale(scale: float) -> None
```

Query or set horizontal time/division in seconds.

---

### get_record_length / set_record_length

```python
scope.get_record_length() -> int
scope.set_record_length(length: int) -> int
```

Query or set horizontal record length. Returns actual length set by scope.

| Parameter | Type | Description |
|-----------|------|-------------|
| `length` | int | Number of points (500, 1000, 2500, 5000, 15000) |

---

### get_horizontal_position / set_horizontal_position

```python
scope.get_horizontal_position() -> float
scope.set_horizontal_position(position: float) -> None
```

Query or set horizontal trigger position in percent.

---

## Trigger Settings

### get_trigger_source / set_trigger_source

```python
scope.get_trigger_source() -> str
scope.set_trigger_source(source: str) -> None
```

Query or set edge trigger source: "CH1"-"CH4", "EXT", or "LINE".

---

### get_trigger_level / set_trigger_level

```python
scope.get_trigger_level() -> float
scope.set_trigger_level(level: float) -> None
```

Query or set trigger level in volts.

---

### get_trigger_slope / set_trigger_slope

```python
scope.get_trigger_slope() -> str
scope.set_trigger_slope(slope: str) -> None
```

Query or set trigger slope: "RISe" or "FALL".

---

### get_trigger_mode / set_trigger_mode

```python
scope.get_trigger_mode() -> str
scope.set_trigger_mode(mode: str) -> None
```

Query or set trigger mode: "AUTO" or "NORMal".

---

## Acquisition Control

### run_acquisition / stop_acquisition

```python
scope.run_acquisition() -> None
scope.stop_acquisition() -> None
```

Start or stop acquisition (live waveform updates).

---

### single_acquisition

```python
scope.single_acquisition() -> None
```

Acquire a single sequence then stop.

---

### get_acquisition_state

```python
scope.get_acquisition_state() -> str
```

Query acquisition state. Returns "0" (stopped) or "1" (running).

---

### get_acquisition_mode / set_acquisition_mode

```python
scope.get_acquisition_mode() -> str
scope.set_acquisition_mode(mode: str) -> None
```

Query or set acquisition mode: "SAMple", "PEAKdetect", "HIRes", "AVErage", or "ENVelope".

---

### get_num_averages / set_num_averages

```python
scope.get_num_averages() -> int
scope.set_num_averages(count: int) -> None
```

Query or set number of waveforms to average.

---

## Waveform Capture

### read_waveform

```python
scope.read_waveform(channel: str) -> WaveformData
```

Read waveform from channel.

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel` | str | Channel name ("CH1", "CH2", "CH3", "CH4") |

**Returns:** `WaveformData` object with `.time`, `.voltage` (numpy arrays), `.preamble` (dict)

**Preamble fields:**
| Field | Type | Description |
|-------|------|-------------|
| `description` | str | Waveform description |
| `nr_pt` | int | Number of points |
| `xincr` | float | Time per point (seconds) |
| `pt_off` | float | Point offset |
| `ymult` | float | Voltage multiplier |
| `yoff` | float | Voltage offset |
| `yzero` | float | Voltage zero offset |

---

## Measurements

### measure_immediate

```python
scope.measure_immediate(channel: str, measurement_type: str) -> float
```

Take an immediate measurement on a channel.

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel` | str | Channel name ("CH1"-"CH4") |
| `measurement_type` | str | See measurement types below |

**Measurement types:**
| Type | Description |
|------|-------------|
| `FREQuency` | Signal frequency |
| `PERIod` | Signal period |
| `PK2pk` | Peak-to-peak voltage |
| `MEAN` | Mean voltage |
| `MINImum` | Minimum voltage |
| `MAXImum` | Maximum voltage |
| `AMPlitude` | Amplitude |
| `RISe` | Rise time |
| `FALL` | Fall time |
| `PWIdth` | Positive pulse width |
| `NWIdth` | Negative pulse width |

---

### configure_measurement_slot / read_measurement_slot

```python
scope.configure_measurement_slot(slot: int, channel: str, measurement_type: str) -> None
scope.read_measurement_slot(slot: int) -> float
```

Configure a persistent measurement slot (1-4) and read its result.

---

## Cursors

### get_cursor_function / set_cursor_function

```python
scope.get_cursor_function() -> str
scope.set_cursor_function(function: str) -> None
```

Query or set cursor type: "HBArs" (voltage), "VBArs" (time), "PAIred", or "OFF".

---

### set_hbar_positions / get_hbar_delta

```python
scope.set_hbar_positions(pos1: float, pos2: float) -> None
scope.get_hbar_delta() -> float
```

Set horizontal bar positions (volts) and query delta.

---

### set_vbar_positions / get_vbar_delta

```python
scope.set_vbar_positions(pos1: float, pos2: float) -> None
scope.get_vbar_delta() -> float
```

Set vertical bar positions (seconds) and query delta.

---

## Error Handling

### check_errors

```python
scope.check_errors() -> str
```

Query scope for errors using ALLEV? command.

**Returns:** Error string or "0,..." if none

---

## Complete Example

```python
from gtape_prologix_drivers import PrologixAdapter, TDS460A

adapter = PrologixAdapter(port="COM4", gpib_address=1)
scope = TDS460A(adapter)

# Identify
print(scope.get_id())

# Configure channel
scope.set_channel_scale("CH1", 1.0)      # 1 V/div
scope.set_channel_coupling("CH1", "DC")

# Configure horizontal
scope.set_horizontal_scale(1e-3)          # 1 ms/div
scope.set_record_length(5000)

# Configure trigger
scope.set_trigger_source("CH1")
scope.set_trigger_level(0.5)
scope.set_trigger_mode("AUTO")

# Capture waveform
scope.run_acquisition()
waveform = scope.read_waveform("CH1")
print(f"Points: {len(waveform.voltage)}")
print(f"Voltage range: {min(waveform.voltage):.3f} to {max(waveform.voltage):.3f} V")

# Take measurements
freq = scope.measure_immediate("CH1", "FREQuency")
pk2pk = scope.measure_immediate("CH1", "PK2pk")
print(f"Frequency: {freq:.2f} Hz, Pk-Pk: {pk2pk:.3f} V")

adapter.close()
```

## Notes

- The TDS460A uses different GPIB commands than the TDS3000 series
- Standard record length is 500-15000 points; Option 05 extends to 120000 points
- Binary transfer via Prologix takes ~30-60 seconds for 15000 points at 115200 baud
