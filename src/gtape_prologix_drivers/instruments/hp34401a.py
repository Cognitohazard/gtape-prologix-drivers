"""HP 34401A Digital Multimeter driver."""

import time
from typing import Optional


class HP34401A:
    """HP 34401A 6.5 digit DMM control class.

    Supports DC/AC voltage (100mV-1000V), DC/AC current (10mA-3A),
    2-wire/4-wire resistance (100Ohm-100MOhm), and frequency measurements.
    """

    def __init__(self, adapter):
        """Initialize HP34401A with adapter."""
        self.adapter = adapter

    def reset(self):
        """Reset multimeter to default state (DC voltage, autorange)."""
        print("[DMM] Resetting HP34401A...")
        if hasattr(self.adapter, 'ser'):
            self.adapter.ser.reset_input_buffer()
        self.adapter.write("*RST")
        time.sleep(2.0)  # HP34401A genuinely needs ~2s to reset hardware
        self.adapter.write("*CLS")
        self.check_errors()

    def _configure(self, scpi_func: str, range_val: float, resolution: Optional[float] = None):
        """Send CONF command with optional resolution."""
        if resolution is not None:
            cmd = f"CONF:{scpi_func} {range_val},{resolution}"
        else:
            cmd = f"CONF:{scpi_func} {range_val}"
        print(f"[DMM] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def set_nplc(self, nplc: float = 1):
        """Set integration time in power line cycles. Valid: 0.02, 0.2, 1, 10, 100.

        Lower NPLC = faster but noisier. NPLC 1 is a good balance (~20ms at 50Hz).
        """
        if nplc not in (0.02, 0.2, 1, 10, 100):
            raise ValueError(f"Invalid NPLC {nplc}. Valid: 0.02, 0.2, 1, 10, 100")
        self.adapter.write(f"VOLT:DC:NPLC {nplc}")

    def get_nplc(self) -> float:
        """Query current NPLC setting."""
        return float(self.adapter.ask("VOLT:DC:NPLC?"))

    def configure_dc_voltage(self, range_volts: float = 10, resolution: Optional[float] = None):
        """Configure for DC voltage measurement. Valid ranges: 0.1, 1, 10, 100, 1000."""
        self._configure("VOLT:DC", range_volts, resolution)

    def configure_ac_voltage(self, range_volts: float = 10, resolution: Optional[float] = None):
        """Configure for AC voltage measurement."""
        self._configure("VOLT:AC", range_volts, resolution)

    def configure_dc_current(self, range_amps: float = 1, resolution: Optional[float] = None):
        """Configure for DC current measurement. Valid ranges: 0.01, 0.1, 1, 3."""
        self._configure("CURR:DC", range_amps, resolution)

    def configure_ac_current(self, range_amps: float = 1, resolution: Optional[float] = None):
        """Configure for AC current measurement."""
        self._configure("CURR:AC", range_amps, resolution)

    def configure_resistance(self, range_ohms: float = 1000, resolution: Optional[float] = None):
        """Configure for 2-wire resistance measurement. Valid ranges: 100-100M."""
        self._configure("RES", range_ohms, resolution)

    def configure_resistance_4wire(self, range_ohms: float = 1000, resolution: Optional[float] = None):
        """Configure for 4-wire resistance measurement."""
        self._configure("FRES", range_ohms, resolution)

    def read(self) -> float:
        """Take a reading using current configuration. Returns float."""
        return float(self.adapter.ask("READ?"))

    def measure_voltage(self, ac: bool = False, range_volts: Optional[float] = None) -> float:
        """DC or AC voltage measurement. Returns volts.

        Args:
            ac: If True, measure AC voltage. Default is DC.
            range_volts: Fixed range (0.1, 1, 10, 100, 1000). None for autorange.
        """
        func = "VOLT:AC" if ac else "VOLT:DC"
        cmd = f"MEAS:{func}? {range_volts}" if range_volts is not None else f"MEAS:{func}?"
        return float(self.adapter.ask(cmd))

    def measure_current(self, ac: bool = False, range_amps: Optional[float] = None) -> float:
        """DC or AC current measurement. Returns amperes.

        Args:
            ac: If True, measure AC current. Default is DC.
            range_amps: Fixed range (0.01, 0.1, 1, 3). None for autorange.
        """
        func = "CURR:AC" if ac else "CURR:DC"
        cmd = f"MEAS:{func}? {range_amps}" if range_amps is not None else f"MEAS:{func}?"
        return float(self.adapter.ask(cmd))

    def measure_resistance(self, four_wire: bool = False, range_ohms: Optional[float] = None) -> float:
        """2-wire or 4-wire resistance measurement. Returns ohms.

        Args:
            four_wire: If True, use 4-wire measurement. Default is 2-wire.
            range_ohms: Fixed range (100 to 100e6). None for autorange.
        """
        func = "FRES" if four_wire else "RES"
        cmd = f"MEAS:{func}? {range_ohms}" if range_ohms is not None else f"MEAS:{func}?"
        return float(self.adapter.ask(cmd))

    def measure_frequency(self, range_volts: Optional[float] = None) -> float:
        """Frequency measurement. Returns Hz.

        Args:
            range_volts: Expected voltage range for frequency input. None for autorange.
        """
        cmd = f"MEAS:FREQ? {range_volts}" if range_volts is not None else "MEAS:FREQ?"
        return float(self.adapter.ask(cmd))

    def check_errors(self):
        """Query multimeter for errors. Returns error string."""
        error = self.adapter.ask("SYST:ERR?")
        if not error.startswith("+0"):
            print(f"[DMM] Error: {error}")
        return error
