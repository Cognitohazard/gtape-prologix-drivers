"""Agilent/Keysight E3631A Triple Output Power Supply driver.

Three outputs: P6V (0-6V, 0-5A), P25V (0-25V, 0-1A), N25V (0 to -25V, 0-1A).
"""

import time


class AgilentE3631A:
    """Agilent E3631A Triple Output Power Supply control class."""

    # Channel definitions
    P6V = 1   # 0-6V, 0-5A
    P25V = 2  # 0-25V, 0-1A
    N25V = 3  # 0 to -25V, 0-1A

    # Channel specifications (voltage_max, current_max)
    CHANNEL_SPECS = {
        P6V: (6.0, 5.0),
        P25V: (25.0, 1.0),
        N25V: (25.0, 1.0)  # Voltage is negative for N25V
    }

    def __init__(self, adapter):
        """Initialize E3631A with adapter."""
        self.adapter = adapter
        self._current_channel = None

    def reset(self):
        """Reset PSU to default state (all outputs 0V, disabled)."""
        print("[PSU] Resetting E3631A...")
        self.adapter.write("*RST")
        time.sleep(1.0)
        self._current_channel = None
        self.check_errors()

    def select_channel(self, channel):
        """Select output channel (P6V=1, P25V=2, N25V=3)."""
        if channel not in self.CHANNEL_SPECS:
            raise ValueError(f"Invalid channel {channel}. Must be 1 (P6V), 2 (P25V), or 3 (N25V)")
        cmd = f"INST:NSEL {channel}"
        print(f"[PSU] {cmd}")
        self.adapter.write(cmd)
        self._current_channel = channel
        self.check_errors()

    def set_voltage(self, voltage):
        """Set voltage for currently selected channel."""
        if self._current_channel is None:
            raise ValueError("No channel selected. Call select_channel() first")

        v_max, _ = self.CHANNEL_SPECS[self._current_channel]

        if self._current_channel == self.N25V:
            if voltage > 0:
                raise ValueError(f"N25V channel requires negative voltage (got {voltage}V)")
            if abs(voltage) > v_max:
                raise ValueError(f"Voltage {voltage}V exceeds N25V range (0 to -{v_max}V)")
        else:
            if voltage < 0 or voltage > v_max:
                raise ValueError(f"Voltage {voltage}V out of range for channel {self._current_channel} (0-{v_max}V)")

        cmd = f"VOLT {voltage}"
        print(f"[PSU] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def set_current_limit(self, current):
        """Set current limit for currently selected channel."""
        if self._current_channel is None:
            raise ValueError("No channel selected. Call select_channel() first")

        _, i_max = self.CHANNEL_SPECS[self._current_channel]

        if current < 0 or current > i_max:
            raise ValueError(f"Current {current}A out of range for channel {self._current_channel} (0-{i_max}A)")

        cmd = f"CURR {current}"
        print(f"[PSU] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def enable_output(self, enable=True):
        """Enable or disable output."""
        cmd = f"OUTP {1 if enable else 0}"
        print(f"[PSU] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def measure_voltage(self):
        """Measure actual output voltage. Returns volts."""
        response = self.adapter.ask("MEAS:VOLT?")
        return float(response)

    def measure_current(self):
        """Measure actual output current. Returns amperes."""
        response = self.adapter.ask("MEAS:CURR?")
        return float(response)

    def check_errors(self):
        """Query PSU for errors. Returns error string."""
        error = self.adapter.ask("SYST:ERR?")
        if not error.startswith("+0"):
            print(f"[PSU] Error: {error}")
        return error

    def configure_output(self, channel, voltage, current_limit):
        """Select channel, set voltage and current limit in one call."""
        self.select_channel(channel)
        self.set_voltage(voltage)
        self.set_current_limit(current_limit)
        print(f"[PSU] Channel {channel} configured: {voltage}V, {current_limit}A limit")
