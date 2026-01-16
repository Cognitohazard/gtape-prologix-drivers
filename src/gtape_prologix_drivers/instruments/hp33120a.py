"""HP33120A Arbitrary Waveform Generator driver.

12-bit DAC (0-2047 valid range), 8-16000 point memory, 40 MSa/s sampling.
"""

import time
import array
import numpy as np


class HP33120A:
    """HP33120A Arbitrary Waveform Generator control class."""

    DAC_MIN = 0
    DAC_MAX = 2047
    MIN_POINTS = 8
    MAX_POINTS = 16000

    def __init__(self, adapter):
        """Initialize HP33120A with adapter."""
        self.adapter = adapter

    def reset(self):
        """Reset AWG to default state."""
        print("[AWG] Resetting HP33120A...")
        self.adapter.write("*RST")
        time.sleep(1.0)
        self.check_errors()

    def upload_waveform(self, waveform_data, name="PULSE"):
        """Upload waveform to volatile memory and copy to named memory.

        Waveform data must be uint16 values in range 0-2047 with 8-16000 points.
        """
        if not isinstance(waveform_data, np.ndarray):
            waveform_data = np.array(waveform_data, dtype=np.uint16)
        else:
            waveform_data = waveform_data.astype(np.uint16)

        num_points = len(waveform_data)
        if num_points < self.MIN_POINTS or num_points > self.MAX_POINTS:
            raise ValueError(f"Waveform must have {self.MIN_POINTS}-{self.MAX_POINTS} points (got {num_points})")

        if np.any(waveform_data < self.DAC_MIN) or np.any(waveform_data > self.DAC_MAX):
            raise ValueError(f"Waveform values must be in range {self.DAC_MIN}-{self.DAC_MAX}")

        # Convert to byte array with MSB-first byte order
        data_array = array.array('H', waveform_data)
        data_array.byteswap()

        print(f"[AWG] Uploading {num_points} point waveform to volatile memory...")
        self.adapter.write_binary("DATA:DAC VOLATILE, ", data_array.tobytes())
        time.sleep(0.5)
        self.check_errors()

        cmd = f"DATA:COPY {name}, VOLATILE"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        time.sleep(2.0)
        self.check_errors()

        print(f"[AWG] Waveform '{name}' uploaded successfully")

    def select_waveform(self, name="PULSE"):
        """Select a waveform from memory."""
        cmd = f"FUNC:USER {name}"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        time.sleep(1.0)
        self.check_errors()

    def set_function_shape_user(self):
        """Set function shape to USER (arbitrary waveform)."""
        cmd = "FUNC:SHAP USER"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

    def configure_output(self, frequency=5000, voltage=0.5, load=50):
        """Configure output frequency, voltage (Vpp), and load impedance."""
        cmd = f"OUTP:LOAD {load}"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

        cmd = f"FREQ {frequency};VOLT {voltage}"
        print(f"[AWG] {cmd}")
        self.adapter.write(cmd)
        self.check_errors()

        print(f"[AWG] Output configured: {frequency}Hz, {voltage}Vpp, {load}Ohm load")

    def check_errors(self):
        """Query AWG for errors. Returns error string."""
        error = self.adapter.ask("SYST:ERR?")
        if not error.startswith("+0"):
            print(f"[AWG] Error: {error}")
        return error

    def setup_arbitrary_waveform(self, waveform_data, name="PULSE",
                                  frequency=5000, voltage=0.5, load=50):
        """Upload, select, and configure arbitrary waveform in one call."""
        self.upload_waveform(waveform_data, name=name)
        self.select_waveform(name=name)
        self.set_function_shape_user()
        self.configure_output(frequency=frequency, voltage=voltage, load=load)
        print("[AWG] Arbitrary waveform setup complete")
