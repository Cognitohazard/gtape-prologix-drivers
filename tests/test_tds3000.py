"""Unit tests for TDS3000 series oscilloscope drivers."""

import pytest
from unittest.mock import Mock, patch, call
import numpy as np
import struct
from gtape_prologix_drivers.instruments.tds3000 import TDS3000Base, TDS3054, TDS3012B, WaveformData


@pytest.fixture
def mock_adapter():
    """Create a mock adapter."""
    mock = Mock()
    mock.ser = Mock()
    mock.ser.in_waiting = 0
    return mock


@pytest.fixture
def tds3054(mock_adapter):
    """Create TDS3054 scope with mock adapter."""
    return TDS3054(mock_adapter)


@pytest.fixture
def tds3012b(mock_adapter):
    """Create TDS3012B scope with mock adapter."""
    return TDS3012B(mock_adapter)


class TestTDS3054:
    """Tests for TDS3054 (4-channel) oscilloscope."""

    def test_initialization(self, tds3054, mock_adapter):
        """Test oscilloscope initialization."""
        assert tds3054.adapter == mock_adapter
        assert tds3054.NUM_CHANNELS == 4

    def test_get_id(self, tds3054, mock_adapter):
        """Test instrument identification query."""
        mock_adapter.read.return_value = "TEKTRONIX,TDS3054,0,CF:91.1CT"

        result = tds3054.get_id()

        mock_adapter.write.assert_called_with("*IDN?")
        assert "TDS3054" in result

    def test_reset(self, tds3054, mock_adapter):
        """Test reset command."""
        tds3054.reset()

        mock_adapter.write.assert_called_with("*RST")

    def test_get_active_channels_all(self, tds3054, mock_adapter):
        """Test detecting all 4 channels active."""
        mock_adapter.read.side_effect = ["1", "1", "1", "1"]

        channels = tds3054.get_active_channels()

        assert mock_adapter.read.call_count == 4
        assert channels == ["CH1", "CH2", "CH3", "CH4"]

    def test_get_active_channels_some(self, tds3054, mock_adapter):
        """Test detecting some channels active."""
        mock_adapter.read.side_effect = ["1", "0", "1", "0"]

        channels = tds3054.get_active_channels()

        assert channels == ["CH1", "CH3"]

    def test_get_active_channels_none(self, tds3054, mock_adapter):
        """Test when no channels are active."""
        mock_adapter.read.side_effect = ["0", "0", "0", "0"]

        channels = tds3054.get_active_channels()

        assert channels == []

    def test_get_active_channels_invalid_response(self, tds3054, mock_adapter):
        """Test handling of invalid channel responses."""
        mock_adapter.read.side_effect = ["1", "invalid", "1", ""]

        channels = tds3054.get_active_channels()

        # Only valid "1" responses count
        assert channels == ["CH1", "CH3"]

    def test_set_channel_display(self, tds3054, mock_adapter):
        """Test turning channel display on/off."""
        tds3054.set_channel_display("CH1", True)
        mock_adapter.write.assert_called_with("SELect:CH1 ON")

        tds3054.set_channel_display("CH2", False)
        mock_adapter.write.assert_called_with("SELect:CH2 OFF")

    def test_set_channel_scale(self, tds3054, mock_adapter):
        """Test setting channel vertical scale."""
        tds3054.set_channel_scale("CH1", 2.0)

        mock_adapter.write.assert_called_with("CH1:SCAle 2.0")

    def test_set_channel_position(self, tds3054, mock_adapter):
        """Test setting channel vertical position."""
        tds3054.set_channel_position("CH1", 1.5)

        mock_adapter.write.assert_called_with("CH1:POSition 1.5")

    def test_set_channel_coupling(self, tds3054, mock_adapter):
        """Test setting channel coupling."""
        tds3054.set_channel_coupling("CH1", "DC")
        mock_adapter.write.assert_called_with("CH1:COUPling DC")

        tds3054.set_channel_coupling("CH2", "AC")
        mock_adapter.write.assert_called_with("CH2:COUPling AC")

    def test_set_record_length(self, tds3054, mock_adapter):
        """Test setting record length."""
        mock_adapter.read.return_value = "10000"

        actual = tds3054.set_record_length(10000)

        # Verify both the set command and query were sent
        write_calls = [str(c) for c in mock_adapter.write.call_args_list]
        assert any("HORizontal:RECOrdlength 10000" in c for c in write_calls)
        assert any("HORizontal:RECOrdlength?" in c for c in write_calls)
        assert actual == 10000

    def test_set_horizontal_scale(self, tds3054, mock_adapter):
        """Test setting horizontal timebase."""
        tds3054.set_horizontal_scale(1e-3)

        mock_adapter.write.assert_called_with("HORizontal:MAIn:SCAle 0.001")

    def test_get_sample_rate(self, tds3054, mock_adapter):
        """Test querying sample rate."""
        mock_adapter.read.return_value = "5.0E9"

        rate = tds3054.get_sample_rate()

        mock_adapter.write.assert_called_with("HORizontal:SAMPLERate?")
        assert rate == 5.0e9

    def test_set_trigger_source(self, tds3054, mock_adapter):
        """Test setting trigger source."""
        tds3054.set_trigger_source("CH1")

        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:SOUrce CH1")

    def test_set_trigger_level(self, tds3054, mock_adapter):
        """Test setting trigger level."""
        tds3054.set_trigger_level(1.5)

        mock_adapter.write.assert_called_with("TRIGger:A:LEVel 1.5")

    def test_set_trigger_slope(self, tds3054, mock_adapter):
        """Test setting trigger slope."""
        tds3054.set_trigger_slope("RISe")
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:SLOpe RISe")

        tds3054.set_trigger_slope("FALL")
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:SLOpe FALL")

    def test_set_trigger_mode(self, tds3054, mock_adapter):
        """Test setting trigger mode."""
        tds3054.set_trigger_mode("AUTO")
        mock_adapter.write.assert_called_with("TRIGger:A:MODe AUTO")

        tds3054.set_trigger_mode("NORMal")
        mock_adapter.write.assert_called_with("TRIGger:A:MODe NORMal")

    def test_force_trigger(self, tds3054, mock_adapter):
        """Test forcing trigger."""
        tds3054.force_trigger()

        mock_adapter.write.assert_called_with("TRIGger:FORCe")


class TestTDS3012B:
    """Tests for TDS3012B (2-channel) oscilloscope."""

    def test_initialization(self, tds3012b, mock_adapter):
        """Test oscilloscope initialization."""
        assert tds3012b.adapter == mock_adapter
        assert tds3012b.NUM_CHANNELS == 2

    def test_get_active_channels_both(self, tds3012b, mock_adapter):
        """Test detecting both channels active."""
        mock_adapter.read.side_effect = ["1", "1"]

        channels = tds3012b.get_active_channels()

        assert mock_adapter.read.call_count == 2
        assert channels == ["CH1", "CH2"]

    def test_get_active_channels_one(self, tds3012b, mock_adapter):
        """Test detecting single channel active."""
        mock_adapter.read.side_effect = ["0", "1"]

        channels = tds3012b.get_active_channels()

        assert channels == ["CH2"]


class TestPreambleParsing:
    """Tests for preamble parsing."""

    def test_parse_preamble_valid(self, tds3054):
        """Test parsing valid TDS3000 preamble."""
        # TDS3000 format: BYT_NR;BIT_NR;ENCDG;BN_FMT;BYT_OR;NR_PT;WFID;PT_FMT;
        #                 XINCR;PT_OFF;XZERO;XUNIT;YMULT;YZERO;YOFF;YUNIT
        preamble_str = '2;16;BIN;RI;MSB;10000;"Ch1, DC coupling, 2.0V/div";Y;1.0E-6;0;0.0;"s";0.01;0.0;-50;"V"'

        preamble = tds3054._parse_preamble(preamble_str)

        assert preamble['byt_nr'] == 2
        assert preamble['bit_nr'] == 16
        assert preamble['encdg'] == 'BIN'
        assert preamble['bn_fmt'] == 'RI'
        assert preamble['byt_or'] == 'MSB'
        assert preamble['nr_pt'] == 10000
        assert preamble['description'] == 'Ch1, DC coupling, 2.0V/div'
        assert preamble['pt_fmt'] == 'Y'
        assert preamble['xincr'] == pytest.approx(1.0e-6)
        assert preamble['pt_off'] == 0.0
        assert preamble['xzero'] == 0.0
        assert preamble['xunit'] == 's'
        assert preamble['ymult'] == pytest.approx(0.01)
        assert preamble['yzero'] == 0.0
        assert preamble['yoff'] == -50.0
        assert preamble['yunit'] == 'V'

    def test_parse_preamble_incomplete(self, tds3054):
        """Test error on incomplete preamble."""
        preamble_str = '2;16;BIN;RI;MSB;10000'

        with pytest.raises(ValueError) as excinfo:
            tds3054._parse_preamble(preamble_str)

        assert "Incomplete preamble" in str(excinfo.value)


class TestWaveformReading:
    """Tests for waveform reading."""

    def test_read_waveform_complete(self, tds3054, mock_adapter):
        """Test complete waveform reading workflow."""
        preamble_str = '2;16;BIN;RI;MSB;8;"Ch1";Y;1.0E-6;0;0.0;"s";0.01;0.0;0;"V"'
        binary_data = struct.pack('>8h', 100, 200, 300, 400, 500, 600, 700, 800)

        # _ask uses read() for preamble only (no record length query)
        mock_adapter.read.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = tds3054.read_waveform('CH1')

        # Verify configuration commands
        write_calls = [str(c) for c in mock_adapter.write.call_args_list]
        assert any("DATa:SOUrce CH1" in c for c in write_calls)
        assert any("DATa:ENCdg RIBinary" in c for c in write_calls)
        assert any("DATa:WIDth 2" in c for c in write_calls)
        assert any("DATa:STARt 1" in c for c in write_calls)
        assert any("WFMPre?" in c for c in write_calls)
        assert any("CURVe?" in c for c in write_calls)

        # Verify result
        assert isinstance(waveform, WaveformData)
        assert waveform.channel == 'CH1'
        assert len(waveform.voltage) == 8
        assert len(waveform.time) == 8

        # Verify voltage conversion: (data - yoff) * ymult + yzero
        expected_voltages = [v * 0.01 for v in [100, 200, 300, 400, 500, 600, 700, 800]]
        np.testing.assert_array_almost_equal(waveform.voltage, expected_voltages)

        # Verify time conversion
        expected_times = [i * 1e-6 for i in range(8)]
        np.testing.assert_array_almost_equal(waveform.time, expected_times)

    def test_read_waveform_with_offsets(self, tds3054, mock_adapter):
        """Test waveform reading with yoff, yzero, and xzero."""
        preamble_str = '2;16;BIN;RI;MSB;4;"Ch1";Y;1.0E-6;0;1.0E-3;"s";0.02;1.5;-100;"V"'
        binary_data = struct.pack('>4h', 100, 200, 300, 400)

        # _ask uses read() for preamble only
        mock_adapter.read.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = tds3054.read_waveform('CH2')

        # voltage = (data - yoff) * ymult + yzero
        # For first point: (100 - (-100)) * 0.02 + 1.5 = 200 * 0.02 + 1.5 = 5.5
        assert waveform.voltage[0] == pytest.approx(5.5)

        # time = (index - pt_off) * xincr + xzero
        # For first point: (0 - 0) * 1e-6 + 1e-3 = 1e-3
        assert waveform.time[0] == pytest.approx(1e-3)

    def test_read_waveform_incomplete_data(self, tds3054, mock_adapter):
        """Test error when incomplete binary data received."""
        preamble_str = '2;16;BIN;RI;MSB;10;"Ch1";Y;1.0E-6;0;0.0;"s";0.01;0.0;0;"V"'
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)  # Only 5 of 10 points

        # _ask uses read() for preamble only
        mock_adapter.read.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        with pytest.raises(ValueError) as excinfo:
            tds3054.read_waveform('CH1')

        assert "Incomplete data" in str(excinfo.value)


class TestMeasurements:
    """Tests for measurement functions."""

    def test_measure_immediate_frequency(self, tds3054, mock_adapter):
        """Test frequency measurement."""
        mock_adapter.read.return_value = "1.0E6"

        result = tds3054.measure_immediate("CH1", "FREQuency")

        write_calls = [str(c) for c in mock_adapter.write.call_args_list]
        assert any("MEASUrement:IMMed:SOUrce1 CH1" in c for c in write_calls)
        assert any("MEASUrement:IMMed:TYPe FREQuency" in c for c in write_calls)
        mock_adapter.write.assert_called_with("MEASUrement:IMMed:VALue?")
        assert result == pytest.approx(1.0e6)

    def test_measure_immediate_pk2pk(self, tds3054, mock_adapter):
        """Test peak-to-peak measurement."""
        mock_adapter.read.return_value = "3.3"

        result = tds3054.measure_immediate("CH2", "PK2pk")

        assert result == pytest.approx(3.3)

    def test_measure_immediate_mean(self, tds3054, mock_adapter):
        """Test mean measurement."""
        mock_adapter.read.return_value = "1.65"

        result = tds3054.measure_immediate("CH1", "MEAN")

        assert result == pytest.approx(1.65)


class TestErrorHandling:
    """Tests for error handling."""

    def test_check_errors_no_error(self, tds3054, mock_adapter):
        """Test error check with no errors."""
        mock_adapter.read.return_value = "0,No error"

        result = tds3054.check_errors()

        mock_adapter.write.assert_called_with("ALLEv?")
        assert result == "0,No error"

    def test_check_errors_with_error(self, tds3054, mock_adapter):
        """Test error check when error exists."""
        mock_adapter.read.return_value = "100,Command error"

        result = tds3054.check_errors()

        assert result == "100,Command error"


class TestWaveformDataClass:
    """Tests for WaveformData dataclass."""

    def test_waveform_data_structure(self):
        """Test WaveformData dataclass creation."""
        time_array = np.array([0.0, 1e-6, 2e-6])
        voltage_array = np.array([0.0, 1.0, 2.0])
        preamble = {'nr_pt': 3, 'xincr': 1e-6, 'ymult': 0.01}

        waveform = WaveformData(
            channel='CH1',
            time=time_array,
            voltage=voltage_array,
            preamble=preamble
        )

        assert waveform.channel == 'CH1'
        np.testing.assert_array_equal(waveform.time, time_array)
        np.testing.assert_array_equal(waveform.voltage, voltage_array)
        assert waveform.preamble['nr_pt'] == 3


class TestAcquisitionControl:
    """Tests for acquisition control methods."""

    def test_run_acquisition(self, tds3054, mock_adapter):
        """Test run_acquisition command."""
        tds3054.run_acquisition()
        mock_adapter.write.assert_called_with("ACQuire:STATE RUN")

    def test_stop_acquisition(self, tds3054, mock_adapter):
        """Test stop_acquisition command."""
        tds3054.stop_acquisition()
        mock_adapter.write.assert_called_with("ACQuire:STATE STOP")

    def test_single_acquisition(self, tds3054, mock_adapter):
        """Test single_acquisition command."""
        tds3054.single_acquisition()
        calls = mock_adapter.write.call_args_list
        assert call("ACQuire:STOPAfter SEQuence") in calls
        assert call("ACQuire:STATE RUN") in calls

    def test_get_acquisition_state(self, tds3054, mock_adapter):
        """Test querying acquisition state."""
        mock_adapter.read.return_value = "1"
        result = tds3054.get_acquisition_state()
        mock_adapter.write.assert_called_with("ACQuire:STATE?")
        assert result == "1"

    def test_get_acquisition_mode(self, tds3054, mock_adapter):
        """Test querying acquisition mode."""
        mock_adapter.read.return_value = "SAMple"
        result = tds3054.get_acquisition_mode()
        mock_adapter.write.assert_called_with("ACQuire:MODe?")
        assert result == "SAMple"

    def test_set_acquisition_mode(self, tds3054, mock_adapter):
        """Test setting acquisition mode."""
        tds3054.set_acquisition_mode("AVErage")
        mock_adapter.write.assert_called_with("ACQuire:MODe AVErage")

    def test_get_num_averages(self, tds3054, mock_adapter):
        """Test querying number of averages."""
        mock_adapter.read.return_value = "64"
        result = tds3054.get_num_averages()
        assert result == 64

    def test_set_num_averages(self, tds3054, mock_adapter):
        """Test setting number of averages."""
        tds3054.set_num_averages(128)
        mock_adapter.write.assert_called_with("ACQuire:NUMAVg 128")


class TestHorizontalSettings:
    """Tests for horizontal settings methods."""

    def test_get_horizontal_scale(self, tds3054, mock_adapter):
        """Test querying horizontal scale."""
        mock_adapter.read.return_value = "1.0E-3"
        result = tds3054.get_horizontal_scale()
        mock_adapter.write.assert_called_with("HORizontal:MAIn:SCAle?")
        assert result == pytest.approx(1e-3)

    def test_set_horizontal_scale(self, tds3054, mock_adapter):
        """Test setting horizontal scale."""
        tds3054.set_horizontal_scale(1e-6)
        mock_adapter.write.assert_called_with("HORizontal:MAIn:SCAle 1e-06")

    def test_get_horizontal_position(self, tds3054, mock_adapter):
        """Test querying horizontal position."""
        mock_adapter.read.return_value = "50"
        result = tds3054.get_horizontal_position()
        mock_adapter.write.assert_called_with("HORizontal:MAIn:POSition?")
        assert result == pytest.approx(50.0)

    def test_set_horizontal_position(self, tds3054, mock_adapter):
        """Test setting horizontal position."""
        tds3054.set_horizontal_position(25.0)
        mock_adapter.write.assert_called_with("HORizontal:MAIn:POSition 25.0")

    def test_get_delay_mode(self, tds3054, mock_adapter):
        """Test querying delay mode."""
        mock_adapter.read.return_value = "1"
        result = tds3054.get_delay_mode()
        mock_adapter.write.assert_called_with("HORizontal:DELay:MODe?")
        assert result is True

    def test_set_delay_mode(self, tds3054, mock_adapter):
        """Test setting delay mode."""
        tds3054.set_delay_mode(True)
        mock_adapter.write.assert_called_with("HORizontal:DELay:MODe ON")


class TestVerticalSettings:
    """Tests for vertical (channel) settings methods."""

    def test_get_channel_scale(self, tds3054, mock_adapter):
        """Test querying channel scale."""
        mock_adapter.read.return_value = "1.0"
        result = tds3054.get_channel_scale("CH1")
        mock_adapter.write.assert_called_with("CH1:SCAle?")
        assert result == pytest.approx(1.0)

    def test_get_channel_offset(self, tds3054, mock_adapter):
        """Test querying channel offset."""
        mock_adapter.read.return_value = "0.5"
        result = tds3054.get_channel_offset("CH1")
        mock_adapter.write.assert_called_with("CH1:OFFSet?")
        assert result == pytest.approx(0.5)

    def test_set_channel_offset(self, tds3054, mock_adapter):
        """Test setting channel offset."""
        tds3054.set_channel_offset("CH1", 1.0)
        mock_adapter.write.assert_called_with("CH1:OFFSet 1.0")

    def test_get_channel_coupling(self, tds3054, mock_adapter):
        """Test querying channel coupling."""
        mock_adapter.read.return_value = "DC"
        result = tds3054.get_channel_coupling("CH1")
        mock_adapter.write.assert_called_with("CH1:COUPling?")
        assert result == "DC"

    def test_get_channel_bandwidth(self, tds3054, mock_adapter):
        """Test querying channel bandwidth."""
        mock_adapter.read.return_value = "FULL"
        result = tds3054.get_channel_bandwidth("CH1")
        mock_adapter.write.assert_called_with("CH1:BANdwidth?")
        assert result == "FULL"

    def test_set_channel_bandwidth(self, tds3054, mock_adapter):
        """Test setting channel bandwidth."""
        tds3054.set_channel_bandwidth("CH1", "20E6")
        mock_adapter.write.assert_called_with("CH1:BANdwidth 20E6")

    def test_get_channel_impedance(self, tds3054, mock_adapter):
        """Test querying channel impedance."""
        mock_adapter.read.return_value = "MEG"
        result = tds3054.get_channel_impedance("CH1")
        mock_adapter.write.assert_called_with("CH1:IMPedance?")
        assert result == "MEG"

    def test_set_channel_impedance(self, tds3054, mock_adapter):
        """Test setting channel impedance."""
        tds3054.set_channel_impedance("CH1", "FIFty")
        mock_adapter.write.assert_called_with("CH1:IMPedance FIFty")

    def test_get_channel_invert(self, tds3054, mock_adapter):
        """Test querying channel invert."""
        mock_adapter.read.return_value = "0"
        result = tds3054.get_channel_invert("CH1")
        mock_adapter.write.assert_called_with("CH1:INVert?")
        assert result is False

    def test_set_channel_invert(self, tds3054, mock_adapter):
        """Test setting channel invert."""
        tds3054.set_channel_invert("CH1", True)
        mock_adapter.write.assert_called_with("CH1:INVert ON")


class TestTriggerSettings:
    """Tests for trigger settings methods."""

    def test_get_trigger_mode(self, tds3054, mock_adapter):
        """Test querying trigger mode."""
        mock_adapter.read.return_value = "AUTO"
        result = tds3054.get_trigger_mode()
        mock_adapter.write.assert_called_with("TRIGger:A:MODe?")
        assert result == "AUTO"

    def test_get_trigger_type(self, tds3054, mock_adapter):
        """Test querying trigger type."""
        mock_adapter.read.return_value = "EDGe"
        result = tds3054.get_trigger_type()
        mock_adapter.write.assert_called_with("TRIGger:A:TYPe?")
        assert result == "EDGe"

    def test_set_trigger_type(self, tds3054, mock_adapter):
        """Test setting trigger type."""
        tds3054.set_trigger_type("PULSe")
        mock_adapter.write.assert_called_with("TRIGger:A:TYPe PULSe")

    def test_get_trigger_level(self, tds3054, mock_adapter):
        """Test querying trigger level."""
        mock_adapter.read.return_value = "1.5"
        result = tds3054.get_trigger_level()
        mock_adapter.write.assert_called_with("TRIGger:A:LEVel?")
        assert result == pytest.approx(1.5)

    def test_get_trigger_source(self, tds3054, mock_adapter):
        """Test querying trigger source."""
        mock_adapter.read.return_value = "CH1"
        result = tds3054.get_trigger_source()
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:SOUrce?")
        assert result == "CH1"

    def test_get_trigger_slope(self, tds3054, mock_adapter):
        """Test querying trigger slope."""
        mock_adapter.read.return_value = "RISe"
        result = tds3054.get_trigger_slope()
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:SLOpe?")
        assert result == "RISe"

    def test_get_trigger_coupling(self, tds3054, mock_adapter):
        """Test querying trigger coupling."""
        mock_adapter.read.return_value = "DC"
        result = tds3054.get_trigger_coupling()
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:COUPling?")
        assert result == "DC"

    def test_set_trigger_coupling(self, tds3054, mock_adapter):
        """Test setting trigger coupling."""
        tds3054.set_trigger_coupling("HFRej")
        mock_adapter.write.assert_called_with("TRIGger:A:EDGe:COUPling HFRej")


class TestMeasurementSlots:
    """Tests for measurement slot methods."""

    def test_configure_measurement_slot(self, tds3054, mock_adapter):
        """Test configuring measurement slot."""
        tds3054.configure_measurement_slot(1, "CH1", "FREQuency")
        calls = mock_adapter.write.call_args_list
        assert call("MEASUrement:MEAS1:SOUrce1 CH1") in calls
        assert call("MEASUrement:MEAS1:TYPe FREQuency") in calls
        assert call("MEASUrement:MEAS1:STATE ON") in calls

    def test_read_measurement_slot(self, tds3054, mock_adapter):
        """Test reading measurement slot."""
        mock_adapter.read.return_value = "1.0E6"
        result = tds3054.read_measurement_slot(2)
        mock_adapter.write.assert_called_with("MEASUrement:MEAS2:VALue?")
        assert result == pytest.approx(1e6)

    def test_disable_measurement_slot(self, tds3054, mock_adapter):
        """Test disabling measurement slot."""
        tds3054.disable_measurement_slot(3)
        mock_adapter.write.assert_called_with("MEASUrement:MEAS3:STATE OFF")


class TestCursorMethods:
    """Tests for cursor methods."""

    def test_get_cursor_function(self, tds3054, mock_adapter):
        """Test querying cursor function."""
        mock_adapter.read.return_value = "HBARs"
        result = tds3054.get_cursor_function()
        mock_adapter.write.assert_called_with("CURSor:FUNCtion?")
        assert result == "HBARs"

    def test_set_cursor_function(self, tds3054, mock_adapter):
        """Test setting cursor function."""
        tds3054.set_cursor_function("VBARs")
        mock_adapter.write.assert_called_with("CURSor:FUNCtion VBARs")

    def test_set_hbar_positions(self, tds3054, mock_adapter):
        """Test setting HBar positions."""
        tds3054.set_hbar_positions(1.0, 2.0)
        calls = mock_adapter.write.call_args_list
        assert call("CURSor:HBARs:POSITION1 1.0") in calls
        assert call("CURSor:HBARs:POSITION2 2.0") in calls

    def test_get_hbar_delta(self, tds3054, mock_adapter):
        """Test getting HBar delta."""
        mock_adapter.read.return_value = "1.5"
        result = tds3054.get_hbar_delta()
        mock_adapter.write.assert_called_with("CURSor:HBARs:DELTa?")
        assert result == pytest.approx(1.5)

    def test_set_vbar_positions(self, tds3054, mock_adapter):
        """Test setting VBar positions."""
        tds3054.set_vbar_positions(1e-6, 2e-6)
        calls = mock_adapter.write.call_args_list
        assert call("CURSor:VBARs:POSITION1 1e-06") in calls
        assert call("CURSor:VBARs:POSITION2 2e-06") in calls

    def test_get_vbar_delta(self, tds3054, mock_adapter):
        """Test getting VBar delta."""
        mock_adapter.read.return_value = "1.0E-6"
        result = tds3054.get_vbar_delta()
        mock_adapter.write.assert_called_with("CURSor:VBARs:DELTa?")
        assert result == pytest.approx(1e-6)


class TestSystemMethods:
    """Tests for system methods."""

    def test_clear_status(self, tds3054, mock_adapter):
        """Test clear status command."""
        tds3054.clear_status()
        mock_adapter.write.assert_called_with("*CLS")

    def test_get_event_status(self, tds3054, mock_adapter):
        """Test querying event status register."""
        mock_adapter.read.return_value = "32"
        result = tds3054.get_event_status()
        mock_adapter.write.assert_called_with("*ESR?")
        assert result == 32

    def test_get_status_byte(self, tds3054, mock_adapter):
        """Test querying status byte register."""
        mock_adapter.read.return_value = "64"
        result = tds3054.get_status_byte()
        mock_adapter.write.assert_called_with("*STB?")
        assert result == 64

    def test_operation_complete(self, tds3054, mock_adapter):
        """Test operation complete command."""
        tds3054.operation_complete()
        mock_adapter.write.assert_called_with("*OPC")

    def test_wait(self, tds3054, mock_adapter):
        """Test wait command."""
        tds3054.wait()
        mock_adapter.write.assert_called_with("*WAI")

    def test_is_busy_true(self, tds3054, mock_adapter):
        """Test busy query when busy."""
        mock_adapter.read.return_value = "1"
        result = tds3054.is_busy()
        mock_adapter.write.assert_called_with("BUSY?")
        assert result is True

    def test_is_busy_false(self, tds3054, mock_adapter):
        """Test busy query when idle."""
        mock_adapter.read.return_value = "0"
        result = tds3054.is_busy()
        assert result is False

    def test_initialize(self, tds3054, mock_adapter):
        """Test initialize command sequence."""
        mock_adapter.read.return_value = "TEKTRONIX,TDS3054,0,CF:91.1CT"
        result = tds3054.initialize()
        calls = mock_adapter.write.call_args_list
        assert call("*CLS") in calls
        assert call("HEADer OFF") in calls
        assert "TDS3054" in result

    def test_autoset(self, tds3054, mock_adapter):
        """Test autoset command."""
        tds3054.autoset()
        mock_adapter.write.assert_called_with("AUTOSet EXECute")

    def test_set_data_start(self, tds3054, mock_adapter):
        """Test setting data start point."""
        tds3054.set_data_start(100)
        mock_adapter.write.assert_called_with("DATa:STARt 100")

    def test_set_data_stop(self, tds3054, mock_adapter):
        """Test setting data stop point."""
        tds3054.set_data_stop(5000)
        mock_adapter.write.assert_called_with("DATa:STOP 5000")
