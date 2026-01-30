# PrologixAdapter

Core GPIB communication adapter for Prologix GPIB-USB controllers.

## Import

```python
from gtape_prologix_drivers import PrologixAdapter
```

## Constructor

```python
PrologixAdapter(port, gpib_address, timeout=6.0, max_retries=3)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `port` | str | Serial port (e.g., "COM4", "/dev/ttyUSB0") |
| `gpib_address` | int | Initial GPIB address (0-30) |
| `timeout` | float | Read timeout in seconds (default: 6.0) |
| `max_retries` | int | Reconnection attempts on serial errors (default: 3) |

## Methods

### write

```python
adapter.write(command: str) -> None
```

Send a SCPI command to the instrument.

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | str | SCPI command string |

**Example:**
```python
adapter.write("*RST")  # Reset instrument
adapter.write("VOLT 5.0")  # Set voltage
```

---

### read

```python
adapter.read() -> str
```

Read response from instrument. Sends `++read eoi` and reads until newline.

**Returns:** Response string (stripped of whitespace)

**Example:**
```python
adapter.write("*IDN?")
response = adapter.read()
print(response)  # "HEWLETT-PACKARD,34401A,..."
```

---

### ask

```python
adapter.ask(command: str) -> str
```

Send query and read response (combines write + read).

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | str | SCPI query command |

**Returns:** Response string

**Example:**
```python
idn = adapter.ask("*IDN?")
voltage = float(adapter.ask("MEAS:VOLT?"))
```

---

### switch_address

```python
adapter.switch_address(gpib_address: int) -> None
```

Change the target GPIB address for subsequent commands.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpib_address` | int | New GPIB address (0-30) |

**Example:**
```python
adapter.switch_address(22)  # Switch to DMM
voltage = dmm.measure_voltage()

adapter.switch_address(5)   # Switch to PSU
psu.enable_output(True)
```

---

### read_binary

```python
adapter.read_binary(expected_bytes: int | None = None,
                    chunk_size: int = 4096,
                    timeout_override: float | None = None) -> bytes
```

Read binary data in IEEE 488.2 block format (`#<n><length><data>`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_bytes` | int \| None | Expected data size (for timeout calculation) |
| `chunk_size` | int | Read chunk size (default: 4096) |
| `timeout_override` | float \| None | Override timeout in seconds |

**Returns:** Binary data as bytes

**Example:**
```python
adapter.write("CURV?")  # Request waveform data
data = adapter.read_binary(expected_bytes=10000)
```

---

### verify_connection

```python
adapter.verify_connection() -> bool
```

Verify Prologix controller is responding.

**Returns:** True if controller responds correctly

**Example:**
```python
if not adapter.verify_connection():
    print("Prologix controller not responding!")
```

---

### write_binary

```python
adapter.write_binary(command: str, data: bytes | list | tuple) -> None
```

Send binary data with IEEE 488.2 block format (#<N><length><data>).

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | str | SCPI command prefix |
| `data` | bytes \| list \| tuple | Binary data to send |

Special characters (LF, CR, ESC, PLUS) are escaped automatically.

**Example:**
```python
# Upload waveform data to AWG
adapter.write_binary("DATA:DAC VOLATILE, ", waveform_bytes)
```

---

### read_line

```python
adapter.read_line() -> str
```

Read a text line response with retry protection. Identical to `read()` but explicitly for line-based text responses.

**Returns:** Response string (stripped of whitespace)

---

### close

```python
adapter.close() -> None
```

Close the serial connection. Safe to call multiple times.

**Example:**
```python
adapter.close()
```

---

## Context Manager

PrologixAdapter supports the context manager protocol:

```python
with PrologixAdapter(port="COM4", gpib_address=22) as adapter:
    dmm = HP34401A(adapter)
    voltage = dmm.measure_voltage()
# Connection automatically closed
```

## Prologix Configuration

The adapter automatically configures the Prologix controller on connection:

| Setting | Value | Description |
|---------|-------|-------------|
| `++mode` | 1 | Controller mode |
| `++auto` | 0 | Manual read mode |
| `++read_tmo_ms` | 4000 | Read timeout (ms) |
| `++eoi` | 1 | Assert EOI with last byte |
| `++eos` | 0 | No EOS character |

## Error Handling

```python
try:
    adapter = PrologixAdapter(port="COM4", gpib_address=22)
except serial.SerialException as e:
    print(f"Could not open port: {e}")
```
