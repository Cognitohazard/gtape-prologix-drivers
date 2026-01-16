# PLZ164W

Kikusui Electronic Load - 165W, CC/CV/CR/CP modes.

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

## Range Constants

| Constant | Description |
|----------|-------------|
| `PLZ164W.CURR_RANGE_LOW` | Low current range (higher resolution) |
| `PLZ164W.CURR_RANGE_MED` | Medium current range |
| `PLZ164W.CURR_RANGE_HIGH` | High current range (higher current) |

---

## Mode Configuration

> **Important:** Always disable input before changing modes!

### configure_cc_mode

```python
load.configure_cc_mode(current: float, current_range: str = None) -> None
```

Configure Constant Current mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current` | float | Target current in amps |
| `current_range` | str \| None | Range constant (auto if None) |

---

### configure_cv_mode

```python
load.configure_cv_mode(voltage: float) -> None
```

Configure Constant Voltage mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `voltage` | float | Target voltage in volts |

---

### configure_cr_mode

```python
load.configure_cr_mode(resistance: float) -> None
```

Configure Constant Resistance mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `resistance` | float | Target resistance in ohms |

---

### configure_cp_mode

```python
load.configure_cp_mode(power: float) -> None
```

Configure Constant Power mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `power` | float | Target power in watts |

---

## Input Control

### enable_input

```python
load.enable_input(enable: bool) -> None
```

Enable or disable the load input.

| Parameter | Type | Description |
|-----------|------|-------------|
| `enable` | bool | True to enable, False to disable |

---

## Measurements

### measure_voltage

```python
load.measure_voltage() -> float
```

Measure input voltage.

**Returns:** Voltage in volts

---

### measure_current

```python
load.measure_current() -> float
```

Measure input current.

**Returns:** Current in amps

---

### measure_power

```python
load.measure_power() -> float
```

Measure input power.

**Returns:** Power in watts

---

## Setpoint Control

### set_current

```python
load.set_current(current: float) -> None
```

Set current setpoint (CC mode).

---

### set_voltage

```python
load.set_voltage(voltage: float) -> None
```

Set voltage setpoint (CV mode).

---

### set_resistance

```python
load.set_resistance(resistance: float) -> None
```

Set resistance setpoint (CR mode).

---

### set_power

```python
load.set_power(power: float) -> None
```

Set power setpoint (CP mode).

---

## Utility Methods

### get_id

```python
load.get_id() -> str
```

Query instrument identification.

---

### reset

```python
load.reset() -> None
```

Reset to default settings.

---

## Complete Example

```python
from gtape_prologix_drivers import PrologixAdapter, PLZ164W

adapter = PrologixAdapter(port="COM4", gpib_address=10)
load = PLZ164W(adapter)

# Configure CC mode at 500mA
load.configure_cc_mode(current=0.5)
load.enable_input(True)

# Monitor
voltage = load.measure_voltage()
current = load.measure_current()
power = load.measure_power()
print(f"V: {voltage:.3f} V, I: {current:.4f} A, P: {power:.3f} W")

# Change to different current
load.set_current(1.0)

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
3. **Minimum voltage** - Some modes require minimum input voltage to regulate
