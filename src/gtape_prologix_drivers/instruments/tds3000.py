"""TDS3000 Series Digital Phosphor Oscilloscope drivers.

Supports TDS3054 (4 channels, 500MHz) and TDS3012B (2 channels, 100MHz).
"""

import struct
import time

import numpy as np
from dataclasses import dataclass


@dataclass
class WaveformData:
    """Container for oscilloscope waveform data."""
    channel: str
    time: np.ndarray
    voltage: np.ndarray
    preamble: dict


class TDS3000Base:
    """Base class for TDS3000 series oscilloscopes.

    Note: TDS3000 series may need delays between query and read.
    Uses _ask() with configurable delay for reliable communication.
    """

    NUM_CHANNELS: int = 4  # Override in subclasses
    QUERY_DELAY: float = 0.1  # Delay between write and read for queries

    def __init__(self, adapter):
        """Initialize TDS3000 with adapter."""
        self.adapter = adapter

    def _ask(self, command: str, delay: float = None) -> str:
        """Send query and read response with delay for TDS3000 compatibility.

        Args:
            command: SCPI query command
            delay: Delay in seconds before reading (default: QUERY_DELAY)

        Returns:
            Response string from scope
        """
        if delay is None:
            delay = self.QUERY_DELAY
        # Flush any stale data in the serial buffer
        self.adapter.ser.reset_input_buffer()
        self.adapter.write(command)
        time.sleep(delay)
        return self.adapter.read()

    def get_id(self) -> str:
        """Query instrument identification."""
        return self._ask("*IDN?")

    def reset(self) -> None:
        """Reset oscilloscope to default settings."""
        self.adapter.write("*RST")
        time.sleep(2.0)

    def get_active_channels(self) -> list[str]:
        """Detect which channels are currently displayed.

        Uses SELect:CH<x>? query (same as TDS460A).
        """
        active_channels = []
        for ch in range(1, self.NUM_CHANNELS + 1):
            channel_name = f"CH{ch}"
            response = self._ask(f"SELect:{channel_name}?")
            try:
                # Response is "0" or "1"
                if response.strip() == "1":
                    active_channels.append(channel_name)
            except (ValueError, AttributeError):
                pass
        print(f"[Scope] Active channels: {active_channels}")
        return active_channels

    def set_channel_display(self, channel: str, on: bool) -> None:
        """Turn channel display on or off."""
        state = "ON" if on else "OFF"
        self.adapter.write(f"{channel}:DISPlay {state}")

    def set_channel_scale(self, channel: str, volts_per_div: float) -> None:
        """Set vertical scale for a channel."""
        self.adapter.write(f"{channel}:SCAle {volts_per_div}")

    def set_channel_position(self, channel: str, divisions: float) -> None:
        """Set vertical position for a channel in divisions from center."""
        self.adapter.write(f"{channel}:POSition {divisions}")

    def set_channel_coupling(self, channel: str, coupling: str) -> None:
        """Set channel coupling: DC, AC, or GND."""
        self.adapter.write(f"{channel}:COUPling {coupling}")

    def set_record_length(self, length: int) -> int:
        """Set horizontal record length. Returns actual length set by scope."""
        print(f"[Scope] Setting record length to {length} points...")
        self.adapter.write(f"HORizontal:RECOrdlength {length}")
        time.sleep(0.5)

        response = self._ask("HORizontal:RECOrdlength?")
        actual_length = int(response)
        print(f"[Scope] Actual record length: {actual_length}")
        return actual_length

    def set_timebase(self, seconds_per_div: float) -> None:
        """Set horizontal timebase in seconds per division."""
        self.adapter.write(f"HORizontal:MAIn:SCAle {seconds_per_div}")

    def get_sample_rate(self) -> float:
        """Query current sample rate in samples per second.

        Note: This command may not be supported on all TDS3000 firmware versions.
        Sample rate can also be calculated from preamble xincr (1/xincr = sample rate).
        """
        response = self._ask("HORizontal:SAMPLERate?")
        return float(response)

    def set_trigger_source(self, source: str) -> None:
        """Set edge trigger source: CH1-CH4, EXT, LINE."""
        self.adapter.write(f"TRIGger:A:EDGe:SOUrce {source}")

    def set_trigger_level(self, volts: float) -> None:
        """Set trigger level in volts."""
        self.adapter.write(f"TRIGger:A:LEVel {volts}")

    def set_trigger_slope(self, slope: str) -> None:
        """Set trigger slope: RISe or FALL."""
        self.adapter.write(f"TRIGger:A:EDGe:SLOpe {slope}")

    def set_trigger_mode(self, mode: str) -> None:
        """Set trigger mode: AUTO, NORMal, or SINGle."""
        self.adapter.write(f"TRIGger:A:MODe {mode}")

    def force_trigger(self) -> None:
        """Force a trigger event immediately."""
        self.adapter.write("TRIGger:FORCe")

    def run(self) -> None:
        """Start acquisition."""
        self.adapter.write("ACQuire:STATE RUN")

    def stop(self) -> None:
        """Stop acquisition."""
        self.adapter.write("ACQuire:STATE STOP")

    def single(self) -> None:
        """Acquire single sequence then stop."""
        self.adapter.write("ACQuire:STOPAfter SEQuence")
        self.adapter.write("ACQuire:STATE RUN")

    def set_acquire_mode(self, mode: str) -> None:
        """Set acquisition mode: SAMple, PEAKdetect, HIRes, AVErage, ENVelope."""
        self.adapter.write(f"ACQuire:MODe {mode}")

    def set_average_count(self, count: int) -> None:
        """Set number of waveforms to average (2-512)."""
        self.adapter.write(f"ACQuire:NUMAVg {count}")

    def _parse_preamble(self, preamble_str: str) -> dict:
        """Parse WFMOutpre? response into dict.

        TDS3000 WFMOutpre? returns semicolon-separated values:
        BYT_NR;BIT_NR;ENCDG;BN_FMT;BYT_OR;NR_PT;WFID;PT_FMT;XINCR;PT_OFF;
        XZERO;XUNIT;YMULT;YZERO;YOFF;YUNIT
        """
        fields = preamble_str.split(';')

        if len(fields) < 16:
            raise ValueError(f"Incomplete preamble: got {len(fields)} fields, expected 16")

        return {
            'byt_nr': int(fields[0]),       # Bytes per point
            'bit_nr': int(fields[1]),       # Bits per point
            'encdg': fields[2],             # Encoding (BIN, ASC)
            'bn_fmt': fields[3],            # Binary format (RI, RP)
            'byt_or': fields[4],            # Byte order (MSB, LSB)
            'nr_pt': int(fields[5]),        # Number of points
            'wfid': fields[6].strip('"'),   # Waveform ID/description
            'pt_fmt': fields[7],            # Point format (Y, ENV)
            'xincr': float(fields[8]),      # Time per point
            'pt_off': float(fields[9]),     # Point offset
            'xzero': float(fields[10]),     # Time of first point
            'xunit': fields[11].strip('"'), # X units (usually "s")
            'ymult': float(fields[12]),     # Voltage multiplier
            'yzero': float(fields[13]),     # Voltage zero
            'yoff': float(fields[14]),      # Voltage offset
            'yunit': fields[15].strip('"'), # Y units (usually "V")
        }

    def read_waveform(self, channel: str, record_length: int = None) -> WaveformData:
        """Read waveform from channel. Returns WaveformData with time, voltage, and metadata."""
        print(f"[Scope] Reading waveform from {channel}...")

        # Configure data source and format (per TDS3000 reference)
        self.adapter.write(f"DATa:SOUrce {channel}")
        self.adapter.write("DATa:ENCdg BINary")  # TDS3000 uses BINary, not RIBinary
        self.adapter.write("DATa:WIDth 2")
        self.adapter.write("DATa:STARt 1")

        if record_length is None:
            response = self._ask("HORizontal:RECOrdlength?")
            actual_record_length = int(response)
        else:
            actual_record_length = record_length

        self.adapter.write(f"DATa:STOP {actual_record_length}")
        print(f"[Scope] Configured to transfer {actual_record_length} points...")

        # Query preamble (TDS3000 uses WFMPre?, not WFMOutpre?)
        preamble_response = self._ask("WFMPre?", delay=0.5)
        print(f"[Scope] Preamble response ({len(preamble_response)} chars): {preamble_response[:100]}...")
        preamble = self._parse_preamble(preamble_response)

        # Query curve data (binary response)
        self.adapter.write("CURVe?")
        expected_bytes = preamble['nr_pt'] * preamble['byt_nr']
        binary_data = self.adapter.read_binary(expected_bytes=expected_bytes)

        if len(binary_data) < expected_bytes:
            raise ValueError(f"Incomplete data ({len(binary_data)} of {expected_bytes} bytes)")

        # Convert to voltage and time arrays
        num_points = len(binary_data) // preamble['byt_nr']

        if preamble['byt_nr'] == 2:
            # 16-bit signed integers, big-endian
            data_values = np.array(struct.unpack('>' + 'h' * num_points, binary_data))
        else:
            # 8-bit signed integers
            data_values = np.array(struct.unpack('b' * num_points, binary_data))

        # voltage = (data - yoff) * ymult + yzero
        voltage_array = ((data_values - preamble['yoff']) * preamble['ymult']) + preamble['yzero']

        # time = (point_index - pt_off) * xincr + xzero
        time_array = (np.arange(preamble['nr_pt']) - preamble['pt_off']) * preamble['xincr'] + preamble['xzero']

        print(f"[Scope] Successfully read {len(voltage_array)} points from {channel}")

        return WaveformData(
            channel=channel,
            time=time_array,
            voltage=voltage_array,
            preamble=preamble
        )

    def measure(self, measurement_type: str, channel: str) -> float:
        """Take an immediate measurement on a channel.

        Args:
            measurement_type: FREQuency, PERIod, MEAN, PK2pk, CRMS, MIN, MAX,
                            RISEtime, FALLtime, PWIdth
            channel: CH1, CH2, etc.

        Returns:
            Measurement value as float.
        """
        self.adapter.write(f"MEASUrement:IMMed:SOUrce1 {channel}")
        self.adapter.write(f"MEASUrement:IMMed:TYPe {measurement_type}")
        response = self._ask("MEASUrement:IMMed:VALue?")
        return float(response)

    def check_errors(self) -> str:
        """Query scope for errors using standard event status."""
        # Read and clear event status register
        esr = self._ask("*ESR?")
        if int(esr) != 0:
            # There's an error, try to get more info
            error_msg = f"ESR={esr}"
            print(f"[Scope] Error: {error_msg}")
            return error_msg
        return "0"


class TDS3054(TDS3000Base):
    """TDS3054 Digital Phosphor Oscilloscope.

    4 channels, 500MHz bandwidth, 5GS/s sample rate.
    """
    NUM_CHANNELS = 4


class TDS3012B(TDS3000Base):
    """TDS3012B Digital Phosphor Oscilloscope.

    2 channels, 100MHz bandwidth, 1.25GS/s sample rate.
    """
    NUM_CHANNELS = 2
