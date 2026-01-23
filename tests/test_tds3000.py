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
        mock_adapter.write.assert_called_with("CH1:DISPlay ON")

        tds3054.set_channel_display("CH2", False)
        mock_adapter.write.assert_called_with("CH2:DISPlay OFF")

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

    def test_set_timebase(self, tds3054, mock_adapter):
        """Test setting horizontal timebase."""
        tds3054.set_timebase(1e-3)

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

    def test_run(self, tds3054, mock_adapter):
        """Test starting acquisition."""
        tds3054.run()

        mock_adapter.write.assert_called_with("ACQuire:STATE RUN")

    def test_stop(self, tds3054, mock_adapter):
        """Test stopping acquisition."""
        tds3054.stop()

        mock_adapter.write.assert_called_with("ACQuire:STATE STOP")

    def test_single(self, tds3054, mock_adapter):
        """Test single acquisition."""
        tds3054.single()

        calls = mock_adapter.write.call_args_list
        assert call("ACQuire:STOPAfter SEQuence") in calls
        assert call("ACQuire:STATE RUN") in calls

    def test_set_acquire_mode(self, tds3054, mock_adapter):
        """Test setting acquisition mode."""
        tds3054.set_acquire_mode("AVErage")

        mock_adapter.write.assert_called_with("ACQuire:MODe AVErage")

    def test_set_average_count(self, tds3054, mock_adapter):
        """Test setting average count."""
        tds3054.set_average_count(64)

        mock_adapter.write.assert_called_with("ACQuire:NUMAVg 64")


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
        assert preamble['wfid'] == 'Ch1, DC coupling, 2.0V/div'
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

        # Verify configuration commands (TDS3000 uses BINary, not RIBinary)
        write_calls = [str(c) for c in mock_adapter.write.call_args_list]
        assert any("DATa:SOUrce CH1" in c for c in write_calls)
        assert any("DATa:ENCdg BINary" in c for c in write_calls)
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

    def test_measure_frequency(self, tds3054, mock_adapter):
        """Test frequency measurement."""
        mock_adapter.read.return_value = "1.0E6"

        result = tds3054.measure("FREQuency", "CH1")

        write_calls = [str(c) for c in mock_adapter.write.call_args_list]
        assert any("MEASUrement:IMMed:SOUrce1 CH1" in c for c in write_calls)
        assert any("MEASUrement:IMMed:TYPe FREQuency" in c for c in write_calls)
        mock_adapter.write.assert_called_with("MEASUrement:IMMed:VALue?")
        assert result == pytest.approx(1.0e6)

    def test_measure_pk2pk(self, tds3054, mock_adapter):
        """Test peak-to-peak measurement."""
        mock_adapter.read.return_value = "3.3"

        result = tds3054.measure("PK2pk", "CH2")

        assert result == pytest.approx(3.3)

    def test_measure_mean(self, tds3054, mock_adapter):
        """Test mean measurement."""
        mock_adapter.read.return_value = "1.65"

        result = tds3054.measure("MEAN", "CH1")

        assert result == pytest.approx(1.65)


class TestErrorHandling:
    """Tests for error handling."""

    def test_check_errors_no_error(self, tds3054, mock_adapter):
        """Test error check with no errors."""
        mock_adapter.read.return_value = "0"

        result = tds3054.check_errors()

        mock_adapter.write.assert_called_with("*ESR?")
        assert result == "0"

    def test_check_errors_with_error(self, tds3054, mock_adapter):
        """Test error check when error exists."""
        mock_adapter.read.return_value = "32"  # Command error bit

        with patch('builtins.print') as mock_print:
            result = tds3054.check_errors()

        assert "32" in result
        print_calls = [str(c) for c in mock_print.call_args_list]
        assert any("Error" in c for c in print_calls)


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
