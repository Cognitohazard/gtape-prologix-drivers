"""Kikusui PLZ164W Electronic Load driver.

Supports CC/CV/CR/CP modes. Specs: 1.5-150V, 0-33A (132A@0V), 165W max.
"""

import time


class PLZ164W:
    """Kikusui PLZ164W Electronic Load control class."""

    # Operating modes
    MODE_CC = "CURR"
    MODE_CV = "VOLT"
    MODE_CR = "RES"
    MODE_CP = "POW"
    MODE_CCCV = "CCCV"
    MODE_CRCV = "CRCV"

    # Range definitions
    CURR_RANGE_LOW = "LOW"
    CURR_RANGE_MED = "MEDIUM"
    CURR_RANGE_HIGH = "HIGH"
    VOLT_RANGE_LOW = "LOW"
    VOLT_RANGE_HIGH = "HIGH"

    # Specifications
    VOLTAGE_MIN = 1.5
    VOLTAGE_MAX = 150.0
    CURRENT_MAX = 33.0
    POWER_MAX = 165.0
    RESISTANCE_MIN = 0.5
    RESISTANCE_MAX = 6000.0

    # Protection action (only "LIM" supported via SCPI; "LOAD OFF" requires front panel)
    PROT_ACTION_LIMIT = "LIM"
    OPP_ACTION_LIMIT = PROT_ACTION_LIMIT  # Legacy alias
    OCP_MAX = 36.29
    OPP_MAX = 181.5

    def __init__(self, adapter):
        """Initialize PLZ164W with adapter."""
        self.adapter = adapter
        self._current_mode: str | None = None

    def reset(self) -> None:
        """Reset electronic load to default state (input disabled)."""
        print("[LOAD] Resetting PLZ164W...")
        self.adapter.write("*RST")
        time.sleep(1.0)
        self._current_mode = None
        self.check_errors()

    def get_identification(self) -> str:
        """Query instrument identification string."""
        return self.adapter.ask("*IDN?")

    # --- Mode Control ---

    def set_mode(self, mode: str) -> None:
        """Set operating mode (MODE_CC, MODE_CV, MODE_CR, or MODE_CP)."""
        valid_modes = [self.MODE_CC, self.MODE_CV, self.MODE_CR, self.MODE_CP]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode}. Must be one of: {', '.join(valid_modes)}")
        cmd = f"SOURce:FUNCtion {mode}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self._current_mode = mode
        self.check_errors()

    def get_mode(self) -> str:
        """Query current operating mode. Returns CURR/VOLT/RES/POW."""
        response = self.adapter.ask("SOURce:FUNCtion?")
        return response.strip()

    # --- Parameter Setters with Validation ---

    def _set_parameter(self, name: str, cmd_base: str, value: float,
                       min_val: float, max_val: float, unit: str) -> None:
        """Validate and send SOURce parameter command."""
        if value < min_val or value > max_val:
            raise ValueError(f"{name} {value}{unit} out of range ({min_val}-{max_val}{unit})")
        cmd = f"{cmd_base} {value}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def set_current(self, current: float) -> None:
        """Set constant current level (0 to 33A)."""
        self._set_parameter("Current", "SOURce:CURRent", current, 0, self.CURRENT_MAX, "A")

    def get_current(self, retries: int = 3) -> float:
        """Query current setting. Returns amperes."""
        return self._query_float("SOURce:CURRent?", retries=retries)

    def set_voltage(self, voltage: float) -> None:
        """Set constant voltage level (1.5 to 150V)."""
        self._set_parameter("Voltage", "SOURce:VOLTage", voltage,
                           self.VOLTAGE_MIN, self.VOLTAGE_MAX, "V")

    def get_voltage(self, retries: int = 3) -> float:
        """Query voltage setting. Returns volts."""
        return self._query_float("SOURce:VOLTage?", retries=retries)

    def set_power(self, power: float) -> None:
        """Set constant power level (0 to 165W)."""
        self._set_parameter("Power", "SOURce:POWer", power, 0, self.POWER_MAX, "W")

    def get_power(self, retries: int = 3) -> float:
        """Query power setting. Returns watts."""
        return self._query_float("SOURce:POWer?", retries=retries)

    def set_resistance(self, resistance: float) -> None:
        """Set constant resistance level (0.5 to 6000 ohms) via conductance."""
        if resistance < self.RESISTANCE_MIN or resistance > self.RESISTANCE_MAX:
            raise ValueError(f"Resistance {resistance}Ohm out of range "
                           f"({self.RESISTANCE_MIN}-{self.RESISTANCE_MAX}Ohm)")
        conductance = 1.0 / resistance
        cmd = f"SOURce:CONDuctance {conductance}"
        print(f"[LOAD] {cmd} (R={resistance}Ohm)")
        self.adapter.write(cmd)
        self.check_errors()

    def get_resistance(self, retries: int = 3) -> float:
        """Query resistance setting. Returns ohms."""
        conductance = self._query_float("SOURce:CONDuctance?", retries=retries)
        return 1.0 / conductance if conductance > 0 else float('inf')

    # --- Range Control ---

    def set_current_range(self, range_mode: str) -> None:
        """Set current range (CURR_RANGE_LOW, CURR_RANGE_MED, or CURR_RANGE_HIGH)."""
        valid_ranges = [self.CURR_RANGE_LOW, self.CURR_RANGE_MED, self.CURR_RANGE_HIGH]
        if range_mode not in valid_ranges:
            raise ValueError(f"Invalid current range {range_mode}. Must be 'LOW', 'MEDIUM', or 'HIGH'")
        cmd = f"SOURce:CURRent:RANGe {range_mode}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def set_voltage_range(self, range_mode: str) -> None:
        """Set voltage range (VOLT_RANGE_LOW or VOLT_RANGE_HIGH)."""
        valid_ranges = [self.VOLT_RANGE_LOW, self.VOLT_RANGE_HIGH]
        if range_mode not in valid_ranges:
            raise ValueError(f"Invalid voltage range {range_mode}. Must be 'LOW' or 'HIGH'")
        cmd = f"SOURce:VOLTage:RANGe {range_mode}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    # --- Input Control ---

    def enable_input(self, enable: bool = True) -> None:
        """Enable or disable load input."""
        cmd = f"INPut {1 if enable else 0}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        # PLZ164W sends null byte after INPut commands - drain it and wait for state to settle
        time.sleep(0.1)
        if self.adapter.ser.in_waiting:
            self.adapter.ser.read(self.adapter.ser.in_waiting)
        time.sleep(0.1)  # Extra delay for state to settle before queries
        self.check_errors()

    def get_input_state(self) -> bool:
        """Query input state. Returns True if enabled."""
        # Drain any stale data before querying
        self.adapter.ser.reset_input_buffer()
        response = self.adapter.ask("INPut?")
        return response.strip() == "1"

    def set_short_mode(self, enable: bool = False) -> None:
        """Enable or disable short circuit mode."""
        cmd = f"INPut:SHORt {1 if enable else 0}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    # --- Protection Settings ---

    def _set_protection(self, name: str, cmd_base: str, value: float, max_val: float, unit: str) -> None:
        """Validate and send protection threshold command."""
        if value < 0 or value > max_val:
            raise ValueError(f"{name} threshold {value}{unit} out of range (0-{max_val}{unit})")
        cmd = f"{cmd_base} {value}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def _set_protection_action(self, name: str, cmd_base: str, action: str) -> None:
        """Validate and send protection action command (only LIM supported via SCPI)."""
        if action != self.PROT_ACTION_LIMIT:
            raise ValueError(f"Invalid {name} action '{action}'. Only 'LIM' supported via SCPI. "
                           "Use front panel (Setup > Protect Action) for LOAD OFF mode.")
        cmd = f"{cmd_base} {action}"
        print(f"[LOAD] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def set_overpower_protection(self, power: float, verify: bool = True) -> None:
        """Set overpower protection (OPP) threshold (0 to 181.5W)."""
        self._set_protection("OPP", "SOURce:POWer:PROTection", power, self.OPP_MAX, "W")
        if verify:
            actual = self.get_overpower_protection()
            if abs(actual - power) > 0.1:
                print(f"[LOAD] WARNING: OPP set to {power}W but reads back {actual}W")

    def get_overpower_protection(self, retries: int = 3) -> float:
        """Query OPP threshold. Returns watts."""
        return self._query_float("SOURce:POWer:PROTection?", retries=retries)

    def set_overpower_protection_action(self, action: str) -> None:
        """Set OPP action (only PROT_ACTION_LIMIT supported via SCPI)."""
        self._set_protection_action("OPP", "SOURce:POWer:PROTection:ACTion", action)

    def get_overpower_protection_action(self) -> str:
        """Query OPP action setting. Returns 'LIM' or 'OFF'."""
        response = self.adapter.ask("SOURce:POWer:PROTection:ACTion?")
        return response.strip()

    def set_overcurrent_protection(self, current: float) -> None:
        """Set overcurrent protection (OCP) threshold (0 to 36.29A)."""
        self._set_protection("OCP", "SOURce:CURRent:PROTection", current, self.OCP_MAX, "A")

    def get_overcurrent_protection(self, retries: int = 3) -> float:
        """Query OCP threshold. Returns amperes."""
        return self._query_float("SOURce:CURRent:PROTection?", retries=retries)

    def set_overcurrent_protection_action(self, action: str) -> None:
        """Set OCP action (only PROT_ACTION_LIMIT supported via SCPI)."""
        self._set_protection_action("OCP", "SOURce:CURRent:PROTection:ACTion", action)

    def get_overcurrent_protection_action(self) -> str:
        """Query OCP action setting. Returns 'LIM' or 'OFF'."""
        response = self.adapter.ask("SOURce:CURRent:PROTection:ACTion?")
        return response.strip()

    def set_undervoltage_protection(self, voltage: float, verify: bool = True) -> None:
        """Set undervoltage protection (UVP) threshold (0 to 150V)."""
        self._set_protection("UVP", "SOURce:VOLTage:PROTection:UNDer", voltage, self.VOLTAGE_MAX, "V")
        if verify:
            actual = self.get_undervoltage_protection()
            print(f"[LOAD] UVP verify: set={voltage}V, readback={actual}V")
            if abs(actual - voltage) > 0.01:
                print(f"[LOAD] WARNING: UVP mismatch!")

    def get_undervoltage_protection(self, retries: int = 3) -> float:
        """Query UVP threshold. Returns volts."""
        return self._query_float("SOURce:VOLTage:PROTection:UNDer?", retries=retries)

    # --- Query Helpers ---

    def _query_float(self, command: str, retries: int = 3) -> float:
        """Query a float value with retry on communication errors."""
        last_error = None
        for _ in range(retries):
            try:
                response = self.adapter.ask(command)
                return float(response)
            except ValueError as e:
                last_error = e
                time.sleep(0.1)  # Retry backoff
        raise ValueError(f"Query '{command}' failed after {retries} attempts: {last_error}")

    # --- Measurements ---

    def measure_voltage(self, retries: int = 3) -> float:
        """Measure actual input voltage. Returns volts."""
        return self._query_float("MEASure:VOLTage?", retries=retries)

    def measure_current(self, retries: int = 3) -> float:
        """Measure actual input current. Returns amperes."""
        return self._query_float("MEASure:CURRent?", retries=retries)

    def measure_power(self, retries: int = 3) -> float:
        """Measure actual dissipated power. Returns watts."""
        return self._query_float("MEASure:POWer?", retries=retries)

    def check_errors(self) -> str:
        """Query electronic load for errors. Returns error string.

        Error format: '+0,"No error"' or '0,"No error"' indicates no error.
        Any other response indicates an error condition.
        """
        error = self.adapter.ask("SYSTem:ERRor?")
        # Normalize: some instruments return "+0", others return "0"
        is_error = not (error.startswith("+0") or error.startswith("0,"))
        if is_error:
            print(f"[LOAD] Error: {error}")
        return error

    # --- Convenience Methods ---

    def configure_cc_mode(self, current: float, current_range: str | None = None) -> None:
        """Configure CC mode: set mode, optionally set range, then set current."""
        self.set_mode(self.MODE_CC)
        if current_range is not None:
            self.set_current_range(current_range)
        self.set_current(current)
        print(f"[LOAD] CC mode configured: {current}A")

    def configure_cv_mode(self, voltage: float, voltage_range: str | None = None) -> None:
        """Configure CV mode: set mode, optionally set range, then set voltage."""
        self.set_mode(self.MODE_CV)
        if voltage_range is not None:
            self.set_voltage_range(voltage_range)
        self.set_voltage(voltage)
        print(f"[LOAD] CV mode configured: {voltage}V")

    def configure_cr_mode(self, resistance: float) -> None:
        """Configure CR mode: set mode, then set resistance."""
        self.set_mode(self.MODE_CR)
        self.set_resistance(resistance)
        print(f"[LOAD] CR mode configured: {resistance}Ohm")

    def configure_cp_mode(self, power: float) -> None:
        """Configure CP mode: set mode, then set power."""
        self.set_mode(self.MODE_CP)
        self.set_power(power)
        print(f"[LOAD] CP mode configured: {power}W")
