"""TDS460A Digital Oscilloscope driver.

4 channels, 16-bit data width, 500-15000 point record length (up to 120k with Option 05).
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
        time.sleep(2.0)  # Scope needs time to reconfigure acquisition memory

        # Query and wait for response (read_line blocks until response arrives)
        self.adapter.write("HORizontal:RECOrdlength?")
        response = self.adapter.read_line()

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

    def read_waveform(self, channel: str) -> WaveformData:
        """Read waveform from channel. Returns WaveformData with time, voltage, and metadata."""
        print(f"[Scope] Reading waveform from {channel}...")

        # Configure data source and format (DATa:STOP defaults to record length)
        self.adapter.write(f"DATa:SOUrce {channel}")
        self.adapter.write("DATa:ENCdg RIBinary")
        self.adapter.write("DATa:WIDth 2")
        self.adapter.write("DATa:STARt 1")

        # Query preamble (tells us actual nr_pt)
        preamble_response = self.adapter.ask("WFMPre?")
        preamble = self._parse_preamble(preamble_response)
        print(f"[Scope] Transferring {preamble['nr_pt']} points from {channel}...")

        # Query curve data (binary response)
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

    # -- Misc --

    def get_id(self) -> str:
        """Query instrument identification string."""
        return self.adapter.ask("*IDN?")

    def autoset(self) -> None:
        """Perform automatic setup (adjusts scales, triggers, etc.)."""
        self.adapter.write("AUTOSet EXECUTE")

    # -- Acquisition --

    def stop_acquisition(self) -> None:
        """Stop acquisition, freezing the current waveform display."""
        self.adapter.write("ACQuire:STATE STOP")

    def run_acquisition(self) -> None:
        """Resume acquisition (live waveform updates)."""
        self.adapter.write("ACQuire:STATE RUN")

    def single_acquisition(self) -> None:
        """Acquire a single sequence then stop."""
        self.adapter.write("ACQuire:STOPAfter SEQuence")
        self.adapter.write("ACQuire:STATE RUN")

    def get_acquisition_state(self) -> str:
        """Query acquisition state. Returns '0' (stopped) or '1' (running)."""
        return self.adapter.ask("ACQuire:STATE?")

    def get_acquisition_mode(self) -> str:
        """Query acquisition mode (SAMple, PEAKdetect, HIRes, AVErage, ENVelope)."""
        return self.adapter.ask("ACQuire:MODe?")

    def set_acquisition_mode(self, mode: str) -> None:
        """Set acquisition mode.

        Args:
            mode: SAMple, PEAKdetect, HIRes, AVErage, or ENVelope
        """
        self.adapter.write(f"ACQuire:MODe {mode}")

    def get_num_averages(self) -> int:
        """Query number of waveforms to average."""
        return int(self.adapter.ask("ACQuire:NUMAVg?"))

    def set_num_averages(self, count: int) -> None:
        """Set number of waveforms to average."""
        self.adapter.write(f"ACQuire:NUMAVg {count}")

    # -- Horizontal --

    def get_horizontal_scale(self) -> float:
        """Query horizontal time/div in seconds."""
        return float(self.adapter.ask("HORizontal:MAIn:SCAle?"))

    def set_horizontal_scale(self, scale: float) -> None:
        """Set horizontal time/div in seconds."""
        self.adapter.write(f"HORizontal:MAIn:SCAle {scale}")

    def get_record_length(self) -> int:
        """Query current record length in points."""
        return int(self.adapter.ask("HORizontal:RECOrdlength?"))

    def get_horizontal_position(self) -> float:
        """Query horizontal trigger position in percent."""
        return float(self.adapter.ask("HORizontal:POSition?"))

    def set_horizontal_position(self, position: float) -> None:
        """Set horizontal trigger position in percent."""
        self.adapter.write(f"HORizontal:POSition {position}")

    # -- Vertical (per channel) --

    def get_channel_scale(self, channel: str) -> float:
        """Query volts/div for a channel."""
        return float(self.adapter.ask(f"{channel}:SCAle?"))

    def set_channel_scale(self, channel: str, scale: float) -> None:
        """Set volts/div for a channel."""
        self.adapter.write(f"{channel}:SCAle {scale}")

    def get_channel_offset(self, channel: str) -> float:
        """Query vertical offset for a channel in volts."""
        return float(self.adapter.ask(f"{channel}:OFFSet?"))

    def set_channel_offset(self, channel: str, offset: float) -> None:
        """Set vertical offset for a channel in volts."""
        self.adapter.write(f"{channel}:OFFSet {offset}")

    def get_channel_coupling(self, channel: str) -> str:
        """Query coupling for a channel (DC, AC, GND)."""
        return self.adapter.ask(f"{channel}:COUPling?")

    def set_channel_coupling(self, channel: str, coupling: str) -> None:
        """Set coupling for a channel (DC, AC, GND)."""
        self.adapter.write(f"{channel}:COUPling {coupling}")

    def get_channel_bandwidth(self, channel: str) -> str:
        """Query bandwidth limit for a channel (FULl, TWEnty, etc.)."""
        return self.adapter.ask(f"{channel}:BANdwidth?")

    def set_channel_bandwidth(self, channel: str, bandwidth: str) -> None:
        """Set bandwidth limit for a channel.

        Args:
            channel: CH1-CH4
            bandwidth: FULl, TWEnty (20MHz), or ONEhundred (100MHz)
        """
        self.adapter.write(f"{channel}:BANdwidth {bandwidth}")

    def set_channel_display(self, channel: str, on: bool) -> None:
        """Turn channel display on or off."""
        state = "ON" if on else "OFF"
        self.adapter.write(f"SELect:{channel} {state}")

    # -- Trigger --

    def get_trigger_source(self) -> str:
        """Query edge trigger source."""
        return self.adapter.ask("TRIGger:MAIn:EDGE:SOUrce?")

    def set_trigger_source(self, source: str) -> None:
        """Set edge trigger source (CH1-CH4, EXT, LINE)."""
        self.adapter.write(f"TRIGger:MAIn:EDGE:SOUrce {source}")

    def get_trigger_level(self) -> float:
        """Query trigger level in volts."""
        return float(self.adapter.ask("TRIGger:MAIn:LEVel?"))

    def set_trigger_level(self, level: float) -> None:
        """Set trigger level in volts."""
        self.adapter.write(f"TRIGger:MAIn:LEVel {level}")

    def get_trigger_slope(self) -> str:
        """Query trigger slope (RISe or FALL)."""
        return self.adapter.ask("TRIGger:MAIn:EDGE:SLOpe?")

    def set_trigger_slope(self, slope: str) -> None:
        """Set trigger slope (RISe or FALL)."""
        self.adapter.write(f"TRIGger:MAIn:EDGE:SLOpe {slope}")

    def get_trigger_mode(self) -> str:
        """Query trigger mode (AUTO or NORMal)."""
        return self.adapter.ask("TRIGger:MAIn:MODe?")

    def set_trigger_mode(self, mode: str) -> None:
        """Set trigger mode (AUTO or NORMal)."""
        self.adapter.write(f"TRIGger:MAIn:MODe {mode}")

    # -- Measurements --

    def measure_immediate(self, channel: str, measurement_type: str) -> float:
        """Take an immediate measurement on a channel.

        Args:
            channel: CH1-CH4
            measurement_type: FREQuency, PERIod, PK2pk, MEAN, MINImum, MAXImum,
                            AMPlitude, RISe, FALL, PWIdth, NWIdth

        Returns:
            Measurement value as float.
        """
        self.adapter.write(f"MEASUrement:IMMed:SOUrce1 {channel}")
        self.adapter.write(f"MEASUrement:IMMed:TYPe {measurement_type}")
        return float(self.adapter.ask("MEASUrement:IMMed:VALue?"))

    def configure_measurement_slot(self, slot: int, channel: str,
                                   measurement_type: str) -> None:
        """Configure a persistent measurement slot (1-4).

        Args:
            slot: Measurement slot number (1-4)
            channel: CH1-CH4
            measurement_type: FREQuency, PERIod, PK2pk, MEAN, etc.
        """
        self.adapter.write(f"MEASUrement:MEAS{slot}:SOUrce {channel}")
        self.adapter.write(f"MEASUrement:MEAS{slot}:TYPe {measurement_type}")
        self.adapter.write(f"MEASUrement:MEAS{slot}:STATE ON")

    def read_measurement_slot(self, slot: int) -> float:
        """Read the result from a measurement slot.

        Args:
            slot: Measurement slot number (1-4)

        Returns:
            Measurement value as float.
        """
        return float(self.adapter.ask(f"MEASUrement:MEAS{slot}:VALue?"))

    # -- Cursors --

    def get_cursor_function(self) -> str:
        """Query current cursor type (HBArs, VBArs, PAIred, OFF)."""
        return self.adapter.ask("CURSor:FUNCtion?")

    def set_cursor_function(self, function: str) -> None:
        """Set cursor type.

        Args:
            function: HBArs (voltage), VBArs (time), PAIred, or OFF
        """
        self.adapter.write(f"CURSor:FUNCtion {function}")

    def set_hbar_positions(self, pos1: float, pos2: float) -> None:
        """Set horizontal bar cursor positions in volts."""
        self.adapter.write(f"CURSor:HBArs:POSITION1 {pos1}")
        self.adapter.write(f"CURSor:HBArs:POSITION2 {pos2}")

    def get_hbar_delta(self) -> float:
        """Query difference between horizontal bar cursors in volts."""
        return float(self.adapter.ask("CURSor:HBArs:DELTa?"))

    def set_vbar_positions(self, pos1: float, pos2: float) -> None:
        """Set vertical bar cursor positions in seconds."""
        self.adapter.write(f"CURSor:VBArs:POSITION1 {pos1}")
        self.adapter.write(f"CURSor:VBArs:POSITION2 {pos2}")

    def get_vbar_delta(self) -> float:
        """Query difference between vertical bar cursors in seconds."""
        return float(self.adapter.ask("CURSor:VBArs:DELTa?"))
