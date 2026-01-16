"""TDS460A Digital Oscilloscope driver.

4 channels, 16-bit data width, 500-15000 point record length.
"""

import struct
import time

import numpy as np
from dataclasses import dataclass


# Preamble field indices (semicolon-separated from WFMPre?)
PREAMBLE_DESCRIPTION = 5
PREAMBLE_NR_PT = 6
PREAMBLE_XUNIT = 8
PREAMBLE_XINCR = 9
PREAMBLE_PT_OFF = 10
PREAMBLE_YUNIT = 11
PREAMBLE_YMULT = 12
PREAMBLE_YOFF = 13
PREAMBLE_YZERO = 14


@dataclass
class WaveformData:
    """Container for oscilloscope waveform data."""
    channel: str
    time: np.ndarray
    voltage: np.ndarray
    preamble: dict


class TDS460A:
    """TDS460A Digital Oscilloscope control class."""

    def __init__(self, adapter):
        """Initialize TDS460A with adapter."""
        self.adapter = adapter

    def get_active_channels(self) -> list[str]:
        """Detect which channels (CH1-CH4) are currently displayed. Returns list of names."""
        active_channels = []
        for ch in range(1, 5):
            channel_name = f"CH{ch}"
            response = self.adapter.ask(f"SELect:{channel_name}?")
            try:
                if int(response) == 1:
                    active_channels.append(channel_name)
            except ValueError:
                pass
        print(f"[Scope] Active channels: {active_channels}")
        return active_channels

    def set_record_length(self, length: int) -> int:
        """Set horizontal record length. Returns actual length set by scope."""
        print(f"[Scope] Setting record length to {length} points...")
        self.adapter.write(f"HORizontal:RECOrdlength {length}")
        time.sleep(0.5)

        response = self.adapter.ask("HORizontal:RECOrdlength?")
        actual_length = int(response)
        print(f"[Scope] Actual record length: {actual_length}")

        if actual_length > 15000:
            print(f"[Scope] WARNING: Large record length ({actual_length} points) - slow serial transfer")

        return actual_length

    def _parse_preamble(self, preamble_str: str) -> dict:
        """Parse semicolon-delimited preamble string into dict."""
        fields = preamble_str.split(';')

        if len(fields) < 15:
            raise ValueError(f"Incomplete preamble: got {len(fields)} fields, expected 15")

        return {
            'description': fields[PREAMBLE_DESCRIPTION].strip('"'),
            'nr_pt': int(fields[PREAMBLE_NR_PT]),
            'xunit': fields[PREAMBLE_XUNIT].strip('"'),
            'xincr': float(fields[PREAMBLE_XINCR]),
            'pt_off': float(fields[PREAMBLE_PT_OFF]),
            'yunit': fields[PREAMBLE_YUNIT].strip('"'),
            'ymult': float(fields[PREAMBLE_YMULT]),
            'yoff': float(fields[PREAMBLE_YOFF]),
            'yzero': float(fields[PREAMBLE_YZERO])
        }

    def read_waveform(self, channel: str, record_length: int = None) -> WaveformData:
        """Read waveform from channel. Returns WaveformData with time, voltage, and metadata."""
        print(f"[Scope] Reading waveform from {channel}...")

        # Configure data source and format
        self.adapter.write(f"DATa:SOUrce {channel}")
        self.adapter.write("DATa:ENCdg RIBinary")
        self.adapter.write("DATa:WIDth 2")
        self.adapter.write("DATa:STARt 1")

        if record_length is None:
            response = self.adapter.ask("HORizontal:RECOrdlength?")
            actual_record_length = int(response)
        else:
            actual_record_length = record_length

        self.adapter.write(f"DATa:STOP {actual_record_length}")
        print(f"[Scope] Configured to transfer all {actual_record_length} points...")

        # Query preamble (text response) using adapter's read_line method
        self.adapter.write("WFMPre?")
        preamble_response = self.adapter.read_line()
        preamble = self._parse_preamble(preamble_response)

        # Query curve data (binary response)
        # Note: read_binary now sends ++read eoi internally
        self.adapter.write("CURVe?")
        expected_bytes = preamble['nr_pt'] * 2
        binary_data = self.adapter.read_binary(expected_bytes=expected_bytes)

        if len(binary_data) < expected_bytes:
            raise ValueError(f"Incomplete data ({len(binary_data)} of {expected_bytes} bytes)")

        # Convert to voltage and time arrays using vectorized operations
        num_points = len(binary_data) // 2
        data_values = np.array(struct.unpack('>' + 'h' * num_points, binary_data))

        voltage_array = ((data_values - preamble['yoff']) * preamble['ymult']) + preamble['yzero']
        time_array = (np.arange(preamble['nr_pt']) - preamble['pt_off']) * preamble['xincr']

        print(f"[Scope] Successfully read {len(voltage_array)} points from {channel}")

        return WaveformData(
            channel=channel,
            time=time_array,
            voltage=voltage_array,
            preamble=preamble
        )

    def check_errors(self) -> str:
        """Query scope for errors. Returns error string."""
        error = self.adapter.ask("ALLEV?")
        if error and not error.startswith("0,"):
            print(f"[Scope] Error: {error}")
        return error
