# PLZ164W

Kikusui Electronic Load - 165W max, 1.5-150V, 0-33A, CC/CV/CR/CP modes.

## Import

```python
from gtape_prologix_drivers import PLZ164W
```

## Constructor

```python
PLZ164W(adapter)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | PrologixAdapter | Connected adapter instance |

## Mode Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PLZ164W.MODE_CC` | "CURR" | Constant Current mode |
| `PLZ164W.MODE_CV` | "VOLT" | Constant Voltage mode |
| `PLZ164W.MODE_CR` | "RES" | Constant Resistance mode |
| `PLZ164W.MODE_CP` | "POW" | Constant Power mode |
| `PLZ164W.MODE_CCCV` | "CCCV" | CC + CV combined mode |
| `PLZ164W.MODE_CRCV` | "CRCV" | CR + CV combined mode |

## Range Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PLZ164W.CURR_RANGE_LOW` | "LOW" | Low current range (higher resolution) |
| `PLZ164W.CURR_RANGE_MED` | "MEDIUM" | Medium current range |
| `PLZ164W.CURR_RANGE_HIGH` | "HIGH" | High current range |
| `PLZ164W.VOLT_RANGE_LOW` | "LOW" | Low voltage range |
| `PLZ164W.VOLT_RANGE_HIGH` | "HIGH" | High voltage range |

## Specification Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `VOLTAGE_MIN` | 1.5 | Minimum voltage (V) |
| `VOLTAGE_MAX` | 150.0 | Maximum voltage (V) |
| `CURRENT_MAX` | 33.0 | Maximum current (A) |
| `POWER_MAX` | 165.0 | Maximum power (W) |
| `RESISTANCE_MIN` | 0.5 | Minimum resistance (Ω) |
| `RESISTANCE_MAX` | 6000.0 | Maximum resistance (Ω) |

---

## Quick Configuration

> **Important:** Always disable input before changing modes!

### configure_cc_mode

```python
load.configure_cc_mode(current: float, current_range: str | None = None) -> None
```

Configure Constant Current mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current` | float | Target current in amps (0-33A) |
| `current_range` | str \| None | Range constant (auto if None) |

---

### configure_cv_mode

```python
load.configure_cv_mode(voltage: float, voltage_range: str | None = None) -> None
```

Configure Constant Voltage mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `voltage` | float | Target voltage in volts (1.5-150V) |
| `voltage_range` | str \| None | Range constant (auto if None) |

---

### configure_cr_mode

```python
load.configure_cr_mode(resistance: float) -> None
```

Configure Constant Resistance mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `resistance` | float | Target resistance in ohms (0.5-6000Ω) |

---

### configure_cp_mode

```python
load.configure_cp_mode(power: float) -> None
```

Configure Constant Power mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `power` | float | Target power in watts (0-165W) |

---

## Mode Control

### set_mode / get_mode

```python
load.set_mode(mode: str) -> None
load.get_mode() -> str
```

Set or query operating mode (MODE_CC, MODE_CV, MODE_CR, MODE_CP).

---

## Input Control

### enable_input

```python
load.enable_input(enable: bool = True) -> None
```

Enable or disable the load input.

| Parameter | Type | Description |
|-----------|------|-------------|
| `enable` | bool | True to enable, False to disable |

---

### get_input_state

```python
load.get_input_state() -> bool
```

Query input state.

**Returns:** True if enabled

---

### set_short_mode

```python
load.set_short_mode(enable: bool = False) -> None
```

Enable or disable short circuit mode.

---

## Setpoint Control

### set_current / get_current

```python
load.set_current(current: float) -> None
load.get_current() -> float
```

Set or query current setpoint (0-33A).

---

### set_voltage / get_voltage

```python
load.set_voltage(voltage: float) -> None
load.get_voltage() -> float
```

Set or query voltage setpoint (1.5-150V).

---

### set_resistance / get_resistance

```python
load.set_resistance(resistance: float) -> None
load.get_resistance() -> float
```

Set or query resistance setpoint (0.5-6000Ω). Internally uses conductance.

---

### set_power / get_power

```python
load.set_power(power: float) -> None
load.get_power() -> float
```

Set or query power setpoint (0-165W).

---

## Range Control

### set_current_range

```python
load.set_current_range(range_mode: str) -> None
```

Set current range: CURR_RANGE_LOW, CURR_RANGE_MED, or CURR_RANGE_HIGH.

---

### set_voltage_range

```python
load.set_voltage_range(range_mode: str) -> None
```

Set voltage range: VOLT_RANGE_LOW or VOLT_RANGE_HIGH.

---

## Measurements

### measure_voltage / measure_current / measure_power

```python
load.measure_voltage() -> float   # Returns volts
load.measure_current() -> float   # Returns amps
load.measure_power() -> float     # Returns watts
```

All measurement methods support a `retries` parameter (default: 3) for reliability.

---

## Protection Settings

### set_overpower_protection / get_overpower_protection

```python
load.set_overpower_protection(power: float, verify: bool = True) -> None
load.get_overpower_protection() -> float
```

Set or query OPP threshold (0-181.5W).

---

### set_overcurrent_protection / get_overcurrent_protection

```python
load.set_overcurrent_protection(current: float) -> None
load.get_overcurrent_protection() -> float
```

Set or query OCP threshold (0-36.29A).

---

### set_undervoltage_protection / get_undervoltage_protection

```python
load.set_undervoltage_protection(voltage: float, verify: bool = True) -> None
load.get_undervoltage_protection() -> float
```

Set or query UVP threshold (0-150V).

---

### Protection Action Methods

```python
load.set_overpower_protection_action(action: str) -> None
load.get_overpower_protection_action() -> str
load.set_overcurrent_protection_action(action: str) -> None
load.get_overcurrent_protection_action() -> str
```

Set or query protection action. Only "LIM" (limit) is supported via SCPI. Use front panel for "LOAD OFF" mode.

---

## Utility Methods

### get_identification

```python
load.get_identification() -> str
```

Query instrument identification string.

---

### reset

```python
load.reset() -> None
```

Reset to default settings (input disabled).

---

### check_errors

```python
load.check_errors() -> str
```

Query for errors.

**Returns:** Error string or "0,No error" if none

---

## Complete Example

```python
from gtape_prologix_drivers import PrologixAdapter, PLZ164W

adapter = PrologixAdapter(port="COM4", gpib_address=10)
load = PLZ164W(adapter)

# Configure CC mode at 500mA with low range
load.configure_cc_mode(current=0.5, current_range=PLZ164W.CURR_RANGE_LOW)
load.enable_input(True)

# Monitor
voltage = load.measure_voltage()
current = load.measure_current()
power = load.measure_power()
print(f"V: {voltage:.3f} V, I: {current:.4f} A, P: {power:.3f} W")

# Change to different current
load.set_current(1.0)

# Set protection
load.set_overpower_protection(100.0)  # 100W OPP

# Switch to CV mode
load.enable_input(False)  # Always disable first!
load.configure_cv_mode(voltage=5.0)
load.enable_input(True)

# Cleanup
load.enable_input(False)
adapter.close()
```

## Important Notes

1. **Always disable input before changing modes** - The PLZ164W requires this
2. **Check power limits** - 165W maximum, derate at high temperatures
3. **Minimum voltage** - Some modes require minimum 1.5V input voltage
4. **Protection actions** - Only "LIM" mode is configurable via SCPI; "LOAD OFF" requires front panel setup
