"""TDS3000 Series Digital Phosphor Oscilloscope drivers.

Supports TDS3012, TDS3014, TDS3032, TDS3034, TDS3052, TDS3054 (and B/C variants).
Reference: Tektronix TDS3000 Series Programmer Manual (071-0381-XX)
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
        self.adapter.write(command)
        time.sleep(delay)
        return self.adapter.read()

    # -- Identification & System --

    def get_id(self) -> str:
        """Query instrument identification."""
        return self._ask("*IDN?")

    def initialize(self) -> str:
        """Initialize scope for remote control.

        Clears errors and disables headers for clean query responses.
        Returns instrument ID string.
        """
        self.adapter.write("*CLS")          # Clear status
        self.adapter.write("HEADer OFF")    # Disable headers on responses
        time.sleep(0.2)
        return self._ask("*IDN?")

    def reset(self) -> None:
        """Reset oscilloscope to default settings."""
        self.adapter.write("*RST")
        time.sleep(2.0)

    def clear_status(self) -> None:
        """Clear status registers."""
        self.adapter.write("*CLS")

    def wait_for_complete(self, timeout: float = 10.0) -> bool:
        """Wait for pending operations to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if completed, False if timed out.
        """
        start = time.time()
        while time.time() - start < timeout:
            response = self._ask("*OPC?")
            if response.strip() == "1":
                return True
            time.sleep(0.1)
        return False

    def check_errors(self) -> str:
        """Query scope for errors using ALLEv? command.

        Returns:
            Error string (e.g., "100,Command error") or "0,No error" if none.
        """
        return self._ask("ALLEv?")

    def get_event_status(self) -> int:
        """Query event status register.

        Returns:
            Integer 0-255 representing ESR bits.
        """
        return int(self._ask("*ESR?"))

    def get_status_byte(self) -> int:
        """Query status byte register.

        Returns:
            Integer 0-255 representing STB bits.
        """
        return int(self._ask("*STB?"))

    def operation_complete(self) -> None:
        """Set operation complete flag when all pending operations finish.

        Sets bit 0 in ESR when done. Use get_event_status() to check.
        """
        self.adapter.write("*OPC")

    def wait(self) -> None:
        """Wait for all pending operations to complete before continuing.

        Use this to ensure sequential execution of commands.
        """
        self.adapter.write("*WAI")

    def is_busy(self) -> bool:
        """Query if scope is busy with an operation.

        Returns:
            True if busy, False if idle.
        """
        response = self._ask("BUSY?")
        return response.strip() == "1"

    # -- Channel Display --

    def get_active_channels(self) -> list[str]:
        """Detect which channels are currently displayed.

        Returns:
            List of active channel names (e.g., ['CH1', 'CH3']).
        """
        active_channels = []
        for ch in range(1, self.NUM_CHANNELS + 1):
            channel_name = f"CH{ch}"
            response = self._ask(f"SELect:{channel_name}?")
            if response and response.strip() == "1":
                active_channels.append(channel_name)
        return active_channels

    def set_channel_display(self, channel: str, on: bool) -> None:
        """Turn channel display on or off.

        Args:
            channel: CH1-CH4
            on: True to enable, False to disable
        """
        state = "ON" if on else "OFF"
        self.adapter.write(f"SELect:{channel} {state}")

    # -- Vertical (per channel) --

    def get_channel_scale(self, channel: str) -> float:
        """Query volts/div for a channel."""
        return float(self._ask(f"{channel}:SCAle?"))

    def set_channel_scale(self, channel: str, volts_per_div: float) -> None:
        """Set vertical scale for a channel."""
        self.adapter.write(f"{channel}:SCAle {volts_per_div}")

    def get_channel_position(self, channel: str) -> float:
        """Query vertical position for a channel in divisions."""
        return float(self._ask(f"{channel}:POSition?"))

    def set_channel_position(self, channel: str, divisions: float) -> None:
        """Set vertical position for a channel in divisions from center (-5 to +5)."""
        self.adapter.write(f"{channel}:POSition {divisions}")

    def get_channel_offset(self, channel: str) -> float:
        """Query vertical offset for a channel in volts."""
        return float(self._ask(f"{channel}:OFFSet?"))

    def set_channel_offset(self, channel: str, offset: float) -> None:
        """Set vertical offset for a channel in volts."""
        self.adapter.write(f"{channel}:OFFSet {offset}")

    def get_channel_coupling(self, channel: str) -> str:
        """Query coupling for a channel (DC, AC, GND)."""
        return self._ask(f"{channel}:COUPling?").strip()

    def set_channel_coupling(self, channel: str, coupling: str) -> None:
        """Set channel coupling.

        Args:
            channel: CH1-CH4
            coupling: DC, AC, or GND
        """
        self.adapter.write(f"{channel}:COUPling {coupling}")

    def get_channel_bandwidth(self, channel: str) -> str:
        """Query bandwidth limit for a channel."""
        return self._ask(f"{channel}:BANdwidth?").strip()

    def set_channel_bandwidth(self, channel: str, bandwidth: str) -> None:
        """Set bandwidth limit for a channel.

        Args:
            channel: CH1-CH4
            bandwidth: FULL, 100E6 (100MHz), or 20E6 (20MHz)
        """
        self.adapter.write(f"{channel}:BANdwidth {bandwidth}")

    def get_channel_impedance(self, channel: str) -> str:
        """Query input impedance for a channel."""
        return self._ask(f"{channel}:IMPedance?").strip()

    def set_channel_impedance(self, channel: str, impedance: str) -> None:
        """Set input impedance for a channel.

        Args:
            channel: CH1-CH4
            impedance: FIFty (50 ohm) or MEG (1M ohm)
        """
        self.adapter.write(f"{channel}:IMPedance {impedance}")

    def get_channel_invert(self, channel: str) -> bool:
        """Query if channel is inverted."""
        return self._ask(f"{channel}:INVert?").strip() == "1"

    def set_channel_invert(self, channel: str, invert: bool) -> None:
        """Set channel inversion.

        Args:
            channel: CH1-CH4
            invert: True to invert, False for normal
        """
        state = "ON" if invert else "OFF"
        self.adapter.write(f"{channel}:INVert {state}")

    def get_channel_deskew(self, channel: str) -> float:
        """Query deskew time for a channel in seconds."""
        return float(self._ask(f"{channel}:DESKew?"))

    def set_channel_deskew(self, channel: str, deskew: float) -> None:
        """Set deskew time for a channel in seconds."""
        self.adapter.write(f"{channel}:DESKew {deskew}")

    # -- Horizontal --

    def get_horizontal_scale(self) -> float:
        """Query horizontal time/div in seconds."""
        return float(self._ask("HORizontal:MAIn:SCAle?"))

    def set_horizontal_scale(self, seconds_per_div: float) -> None:
        """Set horizontal timebase in seconds per division."""
        self.adapter.write(f"HORizontal:MAIn:SCAle {seconds_per_div}")

    def get_horizontal_position(self) -> float:
        """Query horizontal trigger position as percent of record (0-100)."""
        return float(self._ask("HORizontal:MAIn:POSition?"))

    def set_horizontal_position(self, position: float) -> None:
        """Set horizontal trigger position.

        Args:
            position: 0-100 (0% = left edge, 50% = center, 100% = right edge)
        """
        self.adapter.write(f"HORizontal:MAIn:POSition {position}")

    def get_record_length(self) -> int:
        """Query current record length in points."""
        return int(self._ask("HORizontal:RECOrdlength?"))

    def set_record_length(self, length: int) -> int:
        """Set horizontal record length.

        Note: TDS3000 only supports specific lengths (e.g., 500, 10000).

        Args:
            length: Desired record length in points.

        Returns:
            Actual length set by scope (may differ from requested).
        """
        self.adapter.write(f"HORizontal:RECOrdlength {length}")
        time.sleep(0.5)
        return int(self._ask("HORizontal:RECOrdlength?"))

    def get_delay_mode(self) -> bool:
        """Query if delayed timebase (zoom) is enabled."""
        return self._ask("HORizontal:DELay:MODe?").strip() == "1"

    def set_delay_mode(self, on: bool) -> None:
        """Enable or disable delayed timebase (zoom).

        Args:
            on: True to enable, False to disable
        """
        state = "ON" if on else "OFF"
        self.adapter.write(f"HORizontal:DELay:MODe {state}")

    def get_delay_time(self) -> float:
        """Query delay time from trigger point in seconds."""
        return float(self._ask("HORizontal:DELay:TIMe?"))

    def set_delay_time(self, seconds: float) -> None:
        """Set delay time from trigger point in seconds."""
        self.adapter.write(f"HORizontal:DELay:TIMe {seconds}")

    # -- Acquisition --

    def run_acquisition(self) -> None:
        """Start acquisition (live waveform updates)."""
        self.adapter.write("ACQuire:STATE RUN")

    def stop_acquisition(self) -> None:
        """Stop acquisition, freezing the current waveform display."""
        self.adapter.write("ACQuire:STATE STOP")

    def single_acquisition(self) -> None:
        """Acquire a single sequence then stop."""
        self.adapter.write("ACQuire:STOPAfter SEQuence")
        self.adapter.write("ACQuire:STATE RUN")

    def get_acquisition_state(self) -> str:
        """Query acquisition state.

        Returns:
            '0' (stopped) or '1' (running).
        """
        return self._ask("ACQuire:STATE?").strip()

    def get_acquisition_mode(self) -> str:
        """Query acquisition mode.

        Returns:
            SAMple, PEAKdetect, HIRes, AVErage, or ENVelope.
        """
        return self._ask("ACQuire:MODe?").strip()

    def set_acquisition_mode(self, mode: str) -> None:
        """Set acquisition mode.

        Args:
            mode: SAMple, PEAKdetect, HIRes, AVErage, or ENVelope
        """
        self.adapter.write(f"ACQuire:MODe {mode}")

    def get_num_averages(self) -> int:
        """Query number of waveforms to average."""
        return int(self._ask("ACQuire:NUMAVg?"))

    def set_num_averages(self, count: int) -> None:
        """Set number of waveforms to average (2-512)."""
        self.adapter.write(f"ACQuire:NUMAVg {count}")

    def get_num_envelopes(self) -> int:
        """Query number of envelopes to acquire (0 = infinite)."""
        return int(self._ask("ACQuire:NUMEnv?"))

    def set_num_envelopes(self, count: int) -> None:
        """Set number of envelopes to acquire (1-2000, or 0/INFInite)."""
        if count == 0:
            self.adapter.write("ACQuire:NUMEnv INFInite")
        else:
            self.adapter.write(f"ACQuire:NUMEnv {count}")

    def get_num_acquisitions(self) -> int:
        """Query number of acquisitions completed since start."""
        return int(self._ask("ACQuire:NUMACq?"))

    def get_stop_after(self) -> str:
        """Query stop-after mode.

        Returns:
            RUNSTop (continuous) or SEQuence (single-shot).
        """
        return self._ask("ACQuire:STOPAfter?").strip()

    def set_stop_after(self, mode: str) -> None:
        """Set stop-after mode.

        Args:
            mode: RUNSTop (continuous) or SEQuence (single-shot)
        """
        self.adapter.write(f"ACQuire:STOPAfter {mode}")

    # -- Trigger --

    def get_trigger_mode(self) -> str:
        """Query trigger mode.

        Returns:
            AUTO or NORMal.
        """
        return self._ask("TRIGger:A:MODe?").strip()

    def set_trigger_mode(self, mode: str) -> None:
        """Set trigger mode.

        Args:
            mode: AUTO (auto-triggers if no event) or NORMal
        """
        self.adapter.write(f"TRIGger:A:MODe {mode}")

    def get_trigger_type(self) -> str:
        """Query trigger type.

        Returns:
            EDGe, VIDeo, COMM, LOGIc, or PULSe.
        """
        return self._ask("TRIGger:A:TYPe?").strip()

    def set_trigger_type(self, trig_type: str) -> None:
        """Set trigger type.

        Args:
            trig_type: EDGe, VIDeo, COMM, LOGIc, or PULSe
        """
        self.adapter.write(f"TRIGger:A:TYPe {trig_type}")

    def get_trigger_level(self) -> float:
        """Query trigger level in volts."""
        return float(self._ask("TRIGger:A:LEVel?"))

    def set_trigger_level(self, volts: float) -> None:
        """Set trigger level in volts."""
        self.adapter.write(f"TRIGger:A:LEVel {volts}")

    def get_trigger_holdoff(self) -> float:
        """Query trigger holdoff time in seconds."""
        return float(self._ask("TRIGger:A:HOLdoff?"))

    def set_trigger_holdoff(self, seconds: float) -> None:
        """Set trigger holdoff time in seconds."""
        self.adapter.write(f"TRIGger:A:HOLdoff {seconds}")

    def get_trigger_source(self) -> str:
        """Query edge trigger source.

        Returns:
            CH1-CH4, EXT, EXT5, EXT10, LINE, or VERTical.
        """
        return self._ask("TRIGger:A:EDGe:SOUrce?").strip()

    def set_trigger_source(self, source: str) -> None:
        """Set edge trigger source.

        Args:
            source: CH1-CH4, EXT, EXT5, EXT10, LINE, or VERTical
        """
        self.adapter.write(f"TRIGger:A:EDGe:SOUrce {source}")

    def get_trigger_slope(self) -> str:
        """Query trigger slope.

        Returns:
            RISe or FALL.
        """
        return self._ask("TRIGger:A:EDGe:SLOpe?").strip()

    def set_trigger_slope(self, slope: str) -> None:
        """Set trigger slope.

        Args:
            slope: RISe or FALL
        """
        self.adapter.write(f"TRIGger:A:EDGe:SLOpe {slope}")

    def get_trigger_coupling(self) -> str:
        """Query trigger coupling.

        Returns:
            DC, AC, HFRej, LFRej, or NOISErej.
        """
        return self._ask("TRIGger:A:EDGe:COUPling?").strip()

    def set_trigger_coupling(self, coupling: str) -> None:
        """Set trigger coupling.

        Args:
            coupling: DC, AC, HFRej, LFRej, or NOISErej
        """
        self.adapter.write(f"TRIGger:A:EDGe:COUPling {coupling}")

    def force_trigger(self) -> None:
        """Force a trigger event immediately."""
        self.adapter.write("TRIGger:FORCe")

    # -- B-Trigger (Delayed) --

    def get_btrigger_state(self) -> bool:
        """Query if B-trigger is enabled."""
        return self._ask("TRIGger:B:STATe?").strip() == "1"

    def set_btrigger_state(self, on: bool) -> None:
        """Enable or disable B-trigger."""
        state = "ON" if on else "OFF"
        self.adapter.write(f"TRIGger:B:STATe {state}")

    def get_btrigger_time(self) -> float:
        """Query B-trigger delay time in seconds."""
        return float(self._ask("TRIGger:B:TIMe?"))

    def set_btrigger_time(self, seconds: float) -> None:
        """Set B-trigger delay time in seconds."""
        self.adapter.write(f"TRIGger:B:TIMe {seconds}")

    def get_btrigger_events(self) -> int:
        """Query B-trigger event count."""
        return int(self._ask("TRIGger:B:EVEpts?"))

    def set_btrigger_events(self, count: int) -> None:
        """Set B-trigger event count."""
        self.adapter.write(f"TRIGger:B:EVEpts {count}")

    # -- Measurements --

    def measure_immediate(self, channel: str, measurement_type: str) -> float:
        """Take an immediate measurement on a channel.

        Args:
            channel: CH1-CH4 or MATH
            measurement_type: FREQuency, PERIod, MEAN, PK2Pk, CRMs, MIN, MAX,
                            RISE, FALL, PWI (pos width), NWI (neg width),
                            BURst, PHAse

        Returns:
            Measurement value as float. Returns 9.9E37 if invalid/clipped.
        """
        self.adapter.write(f"MEASUrement:IMMed:SOUrce1 {channel}")
        self.adapter.write(f"MEASUrement:IMMed:TYPe {measurement_type}")
        response = self._ask("MEASUrement:IMMed:VALue?")
        return float(response)

    def configure_measurement_slot(self, slot: int, channel: str,
                                   measurement_type: str) -> None:
        """Configure a persistent measurement slot (1-4).

        Args:
            slot: Measurement slot number (1-4)
            channel: CH1-CH4 or MATH
            measurement_type: FREQuency, PERIod, MEAN, PK2Pk, etc.
        """
        self.adapter.write(f"MEASUrement:MEAS{slot}:SOUrce1 {channel}")
        self.adapter.write(f"MEASUrement:MEAS{slot}:TYPe {measurement_type}")
        self.adapter.write(f"MEASUrement:MEAS{slot}:STATE ON")

    def read_measurement_slot(self, slot: int) -> float:
        """Read the result from a measurement slot.

        Args:
            slot: Measurement slot number (1-4)

        Returns:
            Measurement value as float.
        """
        return float(self._ask(f"MEASUrement:MEAS{slot}:VALue?"))

    def disable_measurement_slot(self, slot: int) -> None:
        """Disable a measurement slot.

        Args:
            slot: Measurement slot number (1-4)
        """
        self.adapter.write(f"MEASUrement:MEAS{slot}:STATE OFF")

    # -- Cursors --

    def get_cursor_function(self) -> str:
        """Query current cursor type.

        Returns:
            OFF, HBARs, VBARs, or PAIred.
        """
        return self._ask("CURSor:FUNCtion?").strip()

    def set_cursor_function(self, function: str) -> None:
        """Set cursor type.

        Args:
            function: OFF, HBARs (voltage), VBARs (time), or PAIred
        """
        self.adapter.write(f"CURSor:FUNCtion {function}")

    def get_hbar_position(self, bar: int) -> float:
        """Query horizontal bar cursor position in volts.

        Args:
            bar: 1 or 2
        """
        return float(self._ask(f"CURSor:HBARs:POSITION{bar}?"))

    def set_hbar_positions(self, pos1: float, pos2: float) -> None:
        """Set horizontal bar cursor positions in volts."""
        self.adapter.write(f"CURSor:HBARs:POSITION1 {pos1}")
        self.adapter.write(f"CURSor:HBARs:POSITION2 {pos2}")

    def get_hbar_delta(self) -> float:
        """Query difference between horizontal bar cursors in volts."""
        return float(self._ask("CURSor:HBARs:DELTa?"))

    def get_vbar_position(self, bar: int) -> float:
        """Query vertical bar cursor position in seconds.

        Args:
            bar: 1 or 2
        """
        return float(self._ask(f"CURSor:VBARs:POSITION{bar}?"))

    def set_vbar_positions(self, pos1: float, pos2: float) -> None:
        """Set vertical bar cursor positions in seconds."""
        self.adapter.write(f"CURSor:VBARs:POSITION1 {pos1}")
        self.adapter.write(f"CURSor:VBARs:POSITION2 {pos2}")

    def get_vbar_delta(self) -> float:
        """Query difference between vertical bar cursors in seconds."""
        return float(self._ask("CURSor:VBARs:DELTa?"))

    # -- Waveform Transfer --

    def _parse_preamble(self, preamble_str: str) -> dict:
        """Parse WFMPre? response into dict.

        TDS3000 WFMPre? returns semicolon-separated values:
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

    def read_waveform(self, channel: str) -> WaveformData:
        """Read waveform from channel.

        Args:
            channel: CH1-CH4 or MATH

        Returns:
            WaveformData with time, voltage, and metadata.
        """
        # Configure data source and format
        self.adapter.write(f"DATa:SOUrce {channel}")
        self.adapter.write("DATa:ENCdg RIBinary")  # Signed binary, big-endian
        self.adapter.write("DATa:WIDth 2")         # 16-bit for best resolution
        self.adapter.write("DATa:STARt 1")

        # Query preamble (nr_pt tells us actual point count)
        preamble_response = self._ask("WFMPre?", delay=0.5)
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

        # Voltage = (RawValue - YOff) * YMult + YZero
        voltage_array = ((data_values - preamble['yoff']) * preamble['ymult']) + preamble['yzero']

        # Time = (n - PT_Off) * XIncr + XZero
        time_array = (np.arange(preamble['nr_pt']) - preamble['pt_off']) * preamble['xincr'] + preamble['xzero']

        return WaveformData(
            channel=channel,
            time=time_array,
            voltage=voltage_array,
            preamble=preamble
        )

    def get_sample_rate(self) -> float:
        """Query current sample rate in samples per second.

        Note: Can also be calculated from preamble xincr (1/xincr).
        """
        response = self._ask("HORizontal:SAMPLERate?")
        return float(response)

    def set_data_start(self, start: int) -> None:
        """Set waveform data start point.

        Args:
            start: Start index (1-based, default 1).
        """
        self.adapter.write(f"DATa:STARt {start}")

    def set_data_stop(self, stop: int) -> None:
        """Set waveform data stop point.

        Args:
            stop: Stop index (defaults to record length if not set).
        """
        self.adapter.write(f"DATa:STOP {stop}")

    def autoset(self) -> None:
        """Perform automatic setup (adjusts scales, triggers, etc.)."""
        self.adapter.write("AUTOSet EXECute")
        time.sleep(2.0)  # Autoset takes time

    # -- Hard Copy (Screenshots) --

    def set_hardcopy_format(self, fmt: str) -> None:
        """Set hardcopy image format.

        Args:
            fmt: BMP, PNG, PCX, TIFF, EPSON, or HPGL
        """
        self.adapter.write(f"HARDCopy:FORMat {fmt}")

    def set_hardcopy_layout(self, layout: str) -> None:
        """Set hardcopy orientation.

        Args:
            layout: LANDscape or PORTrait
        """
        self.adapter.write(f"HARDCopy:LAYout {layout}")

    def set_hardcopy_port(self, port: str) -> None:
        """Set hardcopy output port.

        Args:
            port: GPIB, RS232, or ETHernet
        """
        self.adapter.write(f"HARDCopy:PORT {port}")

    def start_hardcopy(self) -> None:
        """Initiate hardcopy (screenshot) dump.

        Call read_binary() immediately after to receive the image data.
        """
        self.adapter.write("HARDCopy:STARt")

    # -- File System & Storage --

    def set_current_directory(self, path: str) -> None:
        """Change current directory on scope.

        Args:
            path: Directory path (e.g., "A:\\")
        """
        self.adapter.write(f'FILESystem:CWD "{path}"')

    def get_directory_listing(self) -> str:
        """Get directory listing of current directory."""
        return self._ask("FILESystem:DIR?")

    def save_setup(self, filepath: str) -> None:
        """Save instrument state to file.

        Args:
            filepath: File path on scope storage
        """
        self.adapter.write(f'SAVe:SETUp "{filepath}"')

    def recall_setup(self, filepath: str) -> None:
        """Recall instrument state from file.

        Args:
            filepath: File path on scope storage
        """
        self.adapter.write(f'RECAll:SETUp "{filepath}"')

    def save_waveform(self, filepath: str) -> None:
        """Save waveform data to CSV/ISF file on scope storage.

        Args:
            filepath: File path on scope storage
        """
        self.adapter.write(f'SAVe:WAVEform "{filepath}"')



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
