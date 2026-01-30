"""HP33120A Arbitrary Waveform Generator driver.

15MHz bandwidth, 12-bit DAC (-2047 to +2047), 8-16000 point memory, 40 MSa/s sampling.
"""

import array
import numpy as np


class HP33120A:
    """HP33120A Arbitrary Waveform Generator control class."""

    # DAC range (signed bipolar)
    DAC_MIN = -2047
    DAC_MAX = 2047
    MIN_POINTS = 8
    MAX_POINTS = 16000

    # Function shape constants
    SHAPE_SINE = "SIN"
    SHAPE_SQUARE = "SQU"
    SHAPE_TRIANGLE = "TRI"
    SHAPE_RAMP = "RAMP"
    SHAPE_NOISE = "NOIS"
    SHAPE_DC = "DC"
    SHAPE_USER = "USER"

    # Voltage unit constants
    UNIT_VPP = "VPP"
    UNIT_VRMS = "VRMS"
    UNIT_DBM = "DBM"

    # Load impedance constants
    LOAD_50_OHM = 50
    LOAD_HIGH_Z = "INF"

    # AM source constants (note: BOTH, not INT)
    AM_SOURCE_BOTH = "BOTH"
    AM_SOURCE_EXTERNAL = "EXT"

    # Generic source constants (for BM, FSK)
    SOURCE_INTERNAL = "INT"
    SOURCE_EXTERNAL = "EXT"

    # Trigger source constants
    TRIGGER_IMMEDIATE = "IMM"
    TRIGGER_EXTERNAL = "EXT"
    TRIGGER_BUS = "BUS"

    # Trigger slope constants
    SLOPE_POSITIVE = "POS"
    SLOPE_NEGATIVE = "NEG"

    # Sweep spacing constants
    SWEEP_LINEAR = "LIN"
    SWEEP_LOGARITHMIC = "LOG"

    def __init__(self, adapter):
        """Initialize HP33120A with adapter."""
        self.adapter = adapter

    # --- Query Helpers ---

    def _wait_for_complete(self) -> None:
        """Wait for pending operations to complete using *OPC? query."""
        self.adapter.ask("*OPC?")

    def _query_float(self, command: str) -> float:
        """Query a float value from the instrument."""
        response = self.adapter.ask(command)
        return float(response)

    def _query_string(self, command: str) -> str:
        """Query a string value from the instrument."""
        response = self.adapter.ask(command)
        return response.strip()

    def _query_bool(self, command: str) -> bool:
        """Query a boolean value (0/1 or OFF/ON) from the instrument."""
        response = self.adapter.ask(command).strip()
        return response in ("1", "ON")

    # --- System Methods ---

    def reset(self) -> None:
        """Reset AWG to default state."""
        print("[AWG] Resetting HP33120A...")
        self.adapter.write("*RST")
        self._wait_for_complete()
        self.check_errors()

    def get_identification(self) -> str:
        """Query instrument identification string (*IDN?)."""
        return self._query_string("*IDN?")

    def self_test(self) -> int:
        """Run self-test and return result (0 = pass, non-zero = fail)."""
        return int(self._query_string("*TST?"))

    def get_version(self) -> str:
        """Query SCPI version string."""
        return self._query_string("SYST:VERS?")

    def beep(self) -> None:
        """Sound the front panel beeper."""
        self.adapter.write("SYST:BEEP")

    def save_state(self, location: int) -> None:
        """Save current state to memory location (0-3). Location 0 is overwritten at power-down."""
        if location < 0 or location > 3:
            raise ValueError(f"State location must be 0-3, got {location}")
        self.adapter.write(f"*SAV {location}")
        self._wait_for_complete()

    def recall_state(self, location: int) -> None:
        """Recall state from memory location (0-3)."""
        if location < 0 or location > 3:
            raise ValueError(f"State location must be 0-3, got {location}")
        self.adapter.write(f"*RCL {location}")
        self._wait_for_complete()

    def delete_state(self, location: int) -> None:
        """Delete state from memory location (0-3)."""
        if location < 0 or location > 3:
            raise ValueError(f"State location must be 0-3, got {location}")
        self.adapter.write(f"MEM:STAT:DEL {location}")

    def check_errors(self) -> str:
        """Query AWG for errors. Returns error string."""
        error = self.adapter.ask("SYST:ERR?")
        if not error.startswith("+0"):
            print(f"[AWG] Error: {error}")
        return error

    # --- Display Methods ---

    def set_display(self, enable: bool) -> None:
        """Enable or disable front panel display."""
        self.adapter.write(f"DISP {'ON' if enable else 'OFF'}")

    def get_display(self) -> bool:
        """Query display state."""
        return self._query_bool("DISP?")

    def set_display_text(self, text: str) -> None:
        """Show custom text on front panel display (max 12 chars)."""
        self.adapter.write(f"DISP:TEXT '{text}'")

    def get_display_text(self) -> str:
        """Query current display text."""
        return self._query_string("DISP:TEXT?")

    def clear_display_text(self) -> None:
        """Clear custom display text and return to normal display."""
        self.adapter.write("DISP:TEXT:CLE")

    # --- Output Parameter Getters/Setters ---

    def set_frequency(self, hz: float) -> None:
        """Set output frequency in Hz."""
        self.adapter.write(f"FREQ {hz}")
        self.check_errors()

    def get_frequency(self) -> float:
        """Query output frequency in Hz."""
        return self._query_float("FREQ?")

    def set_amplitude(self, vpp: float) -> None:
        """Set output amplitude in current voltage unit (default Vpp)."""
        self.adapter.write(f"VOLT {vpp}")
        self.check_errors()

    def get_amplitude(self) -> float:
        """Query output amplitude in current voltage unit."""
        return self._query_float("VOLT?")

    def set_offset(self, volts: float) -> None:
        """Set DC offset voltage."""
        self.adapter.write(f"VOLT:OFFS {volts}")
        self.check_errors()

    def get_offset(self) -> float:
        """Query DC offset voltage."""
        return self._query_float("VOLT:OFFS?")

    def set_function_shape(self, shape: str) -> None:
        """Set function shape (use SHAPE_* constants)."""
        self.adapter.write(f"FUNC:SHAP {shape}")
        self.check_errors()

    def get_function_shape(self) -> str:
        """Query current function shape."""
        return self._query_string("FUNC:SHAP?")

    def set_load_impedance(self, load) -> None:
        """Set expected load impedance (50 or 'INF' for high-Z)."""
        self.adapter.write(f"OUTP:LOAD {load}")
        self.check_errors()

    def get_load_impedance(self) -> str:
        """Query load impedance setting. Returns '50' or 'INF'."""
        return self._query_string("OUTP:LOAD?")

    def set_voltage_unit(self, unit: str) -> None:
        """Set voltage unit (use UNIT_* constants: VPP, VRMS, DBM)."""
        self.adapter.write(f"VOLT:UNIT {unit}")
        self.check_errors()

    def get_voltage_unit(self) -> str:
        """Query voltage unit setting."""
        return self._query_string("VOLT:UNIT?")

    def set_duty_cycle(self, percent: float) -> None:
        """Set duty cycle for square waves (20-80%, frequency dependent)."""
        self.adapter.write(f"PULS:DCYC {percent}")
        self.check_errors()

    def get_duty_cycle(self) -> float:
        """Query duty cycle setting in percent."""
        return self._query_float("PULS:DCYC?")

    def set_sync_output(self, enable: bool) -> None:
        """Enable or disable SYNC output terminal."""
        self.adapter.write(f"OUTP:SYNC {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_sync_output(self) -> bool:
        """Query SYNC output state."""
        return self._query_bool("OUTP:SYNC?")

    # --- APPLy Methods (Standard Waveforms) ---

    def _apply(self, shape: str, freq, amplitude=None, offset=None) -> None:
        """Internal helper for APPLy commands."""
        if offset is not None and amplitude is None:
            raise ValueError("If offset is specified, amplitude must also be specified")
        cmd = f"APPL:{shape} {freq}"
        if amplitude is not None:
            cmd += f", {amplitude}"
            if offset is not None:
                cmd += f", {offset}"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        self._wait_for_complete()
        self.check_errors()

    def apply_sine(self, freq: float, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output sine wave at specified frequency, amplitude (Vpp), and offset."""
        self._apply("SIN", freq, amplitude, offset)

    def apply_square(self, freq: float, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output square wave at specified frequency, amplitude (Vpp), and offset."""
        self._apply("SQU", freq, amplitude, offset)

    def apply_triangle(self, freq: float, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output triangle wave at specified frequency, amplitude (Vpp), and offset."""
        self._apply("TRI", freq, amplitude, offset)

    def apply_ramp(self, freq: float, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output ramp wave at specified frequency, amplitude (Vpp), and offset."""
        self._apply("RAMP", freq, amplitude, offset)

    def apply_noise(self, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output noise with specified amplitude (Vpp) and offset. Frequency param is ignored."""
        self._apply("NOIS", "DEF", amplitude, offset)

    def apply_dc(self, offset: float) -> None:
        """Output DC voltage at specified offset."""
        # DC ignores frequency and amplitude, only offset matters
        self.adapter.write(f"APPL:DC DEF, DEF, {offset}")
        self._wait_for_complete()
        self.check_errors()

    def apply_user(self, freq: float, amplitude: float | None = None, offset: float | None = None) -> None:
        """Output currently selected user waveform."""
        self._apply("USER", freq, amplitude, offset)

    def get_apply_config(self) -> str:
        """Query current APPLy configuration string."""
        return self._query_string("APPL?")

    # --- Arbitrary Waveform Methods ---

    def upload_waveform_dac(self, data, name: str | None = None) -> None:
        """Upload waveform using DAC integer values (-2047 to +2047).

        Args:
            data: Waveform data as list or numpy array of signed integers
            name: If provided, copy to non-volatile memory with this name
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.int16)
        else:
            data = data.astype(np.int16)

        num_points = len(data)
        if num_points < self.MIN_POINTS or num_points > self.MAX_POINTS:
            raise ValueError(f"Waveform must have {self.MIN_POINTS}-{self.MAX_POINTS} points (got {num_points})")

        if np.any(data < self.DAC_MIN) or np.any(data > self.DAC_MAX):
            raise ValueError(f"Waveform values must be in range {self.DAC_MIN} to {self.DAC_MAX}")

        # Convert to byte array with MSB-first byte order (big-endian)
        data_array = array.array('h', data)  # 'h' = signed short
        data_array.byteswap()  # Swap to big-endian

        print(f"[AWG] Uploading {num_points} point waveform to volatile memory...")
        self.adapter.write_binary("DATA:DAC VOLATILE, ", data_array.tobytes())
        self._wait_for_complete()
        self.check_errors()

        if name is not None:
            cmd = f"DATA:COPY {name}, VOLATILE"
            print(f"[AWG] {cmd}")
            self.adapter.write(cmd)
            self._wait_for_complete()
            self.check_errors()
            print(f"[AWG] Waveform '{name}' uploaded successfully")

    def upload_waveform_float(self, data, name: str | None = None) -> None:
        """Upload waveform using normalized float values (-1.0 to +1.0).

        Args:
            data: Waveform data as list or numpy array of floats
            name: If provided, copy to non-volatile memory with this name
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)

        num_points = len(data)
        if num_points < self.MIN_POINTS or num_points > self.MAX_POINTS:
            raise ValueError(f"Waveform must have {self.MIN_POINTS}-{self.MAX_POINTS} points (got {num_points})")

        if np.any(data < -1.0) or np.any(data > 1.0):
            raise ValueError("Waveform values must be in range -1.0 to +1.0")

        # Format as comma-separated values
        values_str = ", ".join(f"{v:.6g}" for v in data)
        cmd = f"DATA VOLATILE, {values_str}"
        print(f"[AWG] Uploading {num_points} point float waveform to volatile memory...")
        self.adapter.write(cmd)
        self._wait_for_complete()
        self.check_errors()

        if name is not None:
            cmd = f"DATA:COPY {name}, VOLATILE"
            print(f"[AWG] {cmd}")
            self.adapter.write(cmd)
            self._wait_for_complete()
            self.check_errors()
            print(f"[AWG] Waveform '{name}' uploaded successfully")

    def select_user_waveform(self, name: str) -> None:
        """Select a waveform from memory for USER output."""
        cmd = f"FUNC:USER {name}"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        self._wait_for_complete()
        self.check_errors()

    def list_waveforms(self) -> str:
        """List all waveforms (built-in and user-defined)."""
        return self._query_string("DATA:CAT?")

    def list_user_waveforms(self) -> str:
        """List user-defined waveforms in non-volatile memory."""
        return self._query_string("DATA:NVOL:CAT?")

    def get_free_memory(self) -> int:
        """Query free non-volatile memory in bytes."""
        return int(self._query_float("DATA:NVOL:FREE?"))

    def get_waveform_points(self, name: str | None = None) -> int:
        """Query number of points in a waveform. If name is None, queries volatile."""
        if name:
            return int(self._query_float(f"DATA:ATTR:POIN? {name}"))
        return int(self._query_float("DATA:ATTR:POIN?"))

    def delete_waveform(self, name: str) -> None:
        """Delete a user-defined waveform from non-volatile memory."""
        self.adapter.write(f"DATA:DEL {name}")
        self._wait_for_complete()
        self.check_errors()

    def delete_all_waveforms(self) -> None:
        """Delete all user-defined waveforms from non-volatile memory."""
        self.adapter.write("DATA:DEL:ALL")
        self._wait_for_complete()
        self.check_errors()

    # --- Legacy Methods (for backwards compatibility) ---

    def upload_waveform(self, waveform_data, name: str = "PULSE") -> None:
        """Upload waveform to volatile memory and copy to named memory.

        DEPRECATED: Use upload_waveform_dac() or upload_waveform_float() instead.

        Note: This method now expects signed values (-2047 to +2047) per the
        HP33120A specification. Legacy code using 0-2047 range should be updated.
        """
        self.upload_waveform_dac(waveform_data, name=name)

    def select_waveform(self, name: str = "PULSE") -> None:
        """Select a waveform from memory.

        DEPRECATED: Use select_user_waveform() instead.
        """
        self.select_user_waveform(name)

    def set_function_shape_user(self) -> None:
        """Set function shape to USER (arbitrary waveform).

        DEPRECATED: Use set_function_shape(SHAPE_USER) instead.
        """
        self.set_function_shape(self.SHAPE_USER)

    def configure_output(self, frequency: float = 5000, voltage: float = 0.5, load=50) -> None:
        """Configure output frequency, voltage (Vpp), and load impedance.

        DEPRECATED: Use individual setters or apply_* methods instead.
        """
        self.set_load_impedance(load)
        self.set_frequency(frequency)
        self.set_amplitude(voltage)
        print(f"[AWG] Output configured: {frequency}Hz, {voltage}Vpp, {load}Ohm load")

    def setup_arbitrary_waveform(self, waveform_data, name: str = "PULSE",
                                  frequency: float = 5000, voltage: float = 0.5, load=50) -> None:
        """Upload, select, and configure arbitrary waveform in one call.

        DEPRECATED: Use upload_waveform_dac() + apply_user() instead.
        """
        self.upload_waveform_dac(waveform_data, name=name)
        self.select_user_waveform(name)
        self.set_function_shape(self.SHAPE_USER)
        self.configure_output(frequency=frequency, voltage=voltage, load=load)
        print("[AWG] Arbitrary waveform setup complete")

    # --- AM Modulation Methods ---

    def set_am_depth(self, percent: float) -> None:
        """Set AM modulation depth (0-120%)."""
        self.adapter.write(f"AM:DEPT {percent}")
        self.check_errors()

    def get_am_depth(self) -> float:
        """Query AM modulation depth in percent."""
        return self._query_float("AM:DEPT?")

    def set_am_source(self, source: str) -> None:
        """Set AM source (AM_SOURCE_BOTH or AM_SOURCE_EXTERNAL)."""
        self.adapter.write(f"AM:SOUR {source}")
        self.check_errors()

    def get_am_source(self) -> str:
        """Query AM source setting."""
        return self._query_string("AM:SOUR?")

    def set_am_internal_function(self, shape: str) -> None:
        """Set AM internal modulating function shape (use SHAPE_* constants, not DC)."""
        self.adapter.write(f"AM:INT:FUNC {shape}")
        self.check_errors()

    def get_am_internal_function(self) -> str:
        """Query AM internal modulating function shape."""
        return self._query_string("AM:INT:FUNC?")

    def set_am_internal_frequency(self, hz: float) -> None:
        """Set AM internal modulating frequency in Hz."""
        self.adapter.write(f"AM:INT:FREQ {hz}")
        self.check_errors()

    def get_am_internal_frequency(self) -> float:
        """Query AM internal modulating frequency in Hz."""
        return self._query_float("AM:INT:FREQ?")

    def set_am_state(self, enable: bool) -> None:
        """Enable or disable AM modulation."""
        self.adapter.write(f"AM:STAT {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_am_state(self) -> bool:
        """Query AM modulation state."""
        return self._query_bool("AM:STAT?")

    # --- FM Modulation Methods ---

    def set_fm_deviation(self, hz: float) -> None:
        """Set FM peak frequency deviation in Hz."""
        self.adapter.write(f"FM:DEV {hz}")
        self.check_errors()

    def get_fm_deviation(self) -> float:
        """Query FM deviation in Hz."""
        return self._query_float("FM:DEV?")

    def set_fm_internal_function(self, shape: str) -> None:
        """Set FM internal modulating function shape (use SHAPE_* constants, not DC)."""
        self.adapter.write(f"FM:INT:FUNC {shape}")
        self.check_errors()

    def get_fm_internal_function(self) -> str:
        """Query FM internal modulating function shape."""
        return self._query_string("FM:INT:FUNC?")

    def set_fm_internal_frequency(self, hz: float) -> None:
        """Set FM internal modulating frequency in Hz."""
        self.adapter.write(f"FM:INT:FREQ {hz}")
        self.check_errors()

    def get_fm_internal_frequency(self) -> float:
        """Query FM internal modulating frequency in Hz."""
        return self._query_float("FM:INT:FREQ?")

    def set_fm_state(self, enable: bool) -> None:
        """Enable or disable FM modulation."""
        self.adapter.write(f"FM:STAT {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_fm_state(self) -> bool:
        """Query FM modulation state."""
        return self._query_bool("FM:STAT?")

    # --- Burst Modulation Methods ---

    def set_burst_cycles(self, count) -> None:
        """Set burst cycle count (integer or 'INF' for infinite)."""
        self.adapter.write(f"BM:NCYC {count}")
        self.check_errors()

    def get_burst_cycles(self) -> str:
        """Query burst cycle count. Returns number or 'INF'."""
        return self._query_string("BM:NCYC?")

    def set_burst_phase(self, degrees: float) -> None:
        """Set burst starting phase in degrees."""
        self.adapter.write(f"BM:PHAS {degrees}")
        self.check_errors()

    def get_burst_phase(self) -> float:
        """Query burst starting phase in degrees."""
        return self._query_float("BM:PHAS?")

    def set_burst_internal_rate(self, hz: float) -> None:
        """Set internal burst rate in Hz."""
        self.adapter.write(f"BM:INT:RATE {hz}")
        self.check_errors()

    def get_burst_internal_rate(self) -> float:
        """Query internal burst rate in Hz."""
        return self._query_float("BM:INT:RATE?")

    def set_burst_source(self, source: str) -> None:
        """Set burst trigger source (SOURCE_INTERNAL or SOURCE_EXTERNAL)."""
        self.adapter.write(f"BM:SOUR {source}")
        self.check_errors()

    def get_burst_source(self) -> str:
        """Query burst trigger source."""
        return self._query_string("BM:SOUR?")

    def set_burst_state(self, enable: bool) -> None:
        """Enable or disable burst modulation."""
        self.adapter.write(f"BM:STAT {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_burst_state(self) -> bool:
        """Query burst modulation state."""
        return self._query_bool("BM:STAT?")

    # --- FSK Methods ---

    def set_fsk_frequency(self, hz: float) -> None:
        """Set FSK hop frequency in Hz."""
        self.adapter.write(f"FSK:FREQ {hz}")
        self.check_errors()

    def get_fsk_frequency(self) -> float:
        """Query FSK hop frequency in Hz."""
        return self._query_float("FSK:FREQ?")

    def set_fsk_internal_rate(self, hz: float) -> None:
        """Set FSK internal shift rate in Hz."""
        self.adapter.write(f"FSK:INT:RATE {hz}")
        self.check_errors()

    def get_fsk_internal_rate(self) -> float:
        """Query FSK internal shift rate in Hz."""
        return self._query_float("FSK:INT:RATE?")

    def set_fsk_source(self, source: str) -> None:
        """Set FSK source (SOURCE_INTERNAL or SOURCE_EXTERNAL)."""
        self.adapter.write(f"FSK:SOUR {source}")
        self.check_errors()

    def get_fsk_source(self) -> str:
        """Query FSK source setting."""
        return self._query_string("FSK:SOUR?")

    def set_fsk_state(self, enable: bool) -> None:
        """Enable or disable FSK."""
        self.adapter.write(f"FSK:STAT {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_fsk_state(self) -> bool:
        """Query FSK state."""
        return self._query_bool("FSK:STAT?")

    # --- Sweep Methods ---

    def set_sweep_start_frequency(self, hz: float) -> None:
        """Set sweep start frequency in Hz."""
        self.adapter.write(f"FREQ:STAR {hz}")
        self.check_errors()

    def get_sweep_start_frequency(self) -> float:
        """Query sweep start frequency in Hz."""
        return self._query_float("FREQ:STAR?")

    def set_sweep_stop_frequency(self, hz: float) -> None:
        """Set sweep stop frequency in Hz."""
        self.adapter.write(f"FREQ:STOP {hz}")
        self.check_errors()

    def get_sweep_stop_frequency(self) -> float:
        """Query sweep stop frequency in Hz."""
        return self._query_float("FREQ:STOP?")

    def set_sweep_spacing(self, spacing: str) -> None:
        """Set sweep spacing (SWEEP_LINEAR or SWEEP_LOGARITHMIC)."""
        self.adapter.write(f"SWE:SPAC {spacing}")
        self.check_errors()

    def get_sweep_spacing(self) -> str:
        """Query sweep spacing setting."""
        return self._query_string("SWE:SPAC?")

    def set_sweep_time(self, seconds: float) -> None:
        """Set sweep time in seconds."""
        self.adapter.write(f"SWE:TIME {seconds}")
        self.check_errors()

    def get_sweep_time(self) -> float:
        """Query sweep time in seconds."""
        return self._query_float("SWE:TIME?")

    def set_sweep_state(self, enable: bool) -> None:
        """Enable or disable frequency sweep."""
        self.adapter.write(f"SWE:STAT {'ON' if enable else 'OFF'}")
        self.check_errors()

    def get_sweep_state(self) -> bool:
        """Query sweep state."""
        return self._query_bool("SWE:STAT?")

    # --- Trigger Methods ---

    def set_trigger_source(self, source: str) -> None:
        """Set trigger source (TRIGGER_IMMEDIATE, TRIGGER_EXTERNAL, or TRIGGER_BUS)."""
        self.adapter.write(f"TRIG:SOUR {source}")
        self.check_errors()

    def get_trigger_source(self) -> str:
        """Query trigger source setting."""
        return self._query_string("TRIG:SOUR?")

    def set_trigger_slope(self, slope: str) -> None:
        """Set trigger slope (SLOPE_POSITIVE or SLOPE_NEGATIVE)."""
        self.adapter.write(f"TRIG:SLOP {slope}")
        self.check_errors()

    def get_trigger_slope(self) -> str:
        """Query trigger slope setting."""
        return self._query_string("TRIG:SLOP?")

    def trigger(self) -> None:
        """Send software trigger (*TRG). Only works when trigger source is BUS."""
        self.adapter.write("*TRG")
