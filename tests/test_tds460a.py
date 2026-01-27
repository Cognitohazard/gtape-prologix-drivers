"""Unit tests for TDS460A oscilloscope driver."""

import pytest
from unittest.mock import patch
import numpy as np
import struct
from gtape_prologix_drivers.instruments.tds460a import TDS460A, WaveformData


class TestTDS460A:
    """Test cases for TDS460A oscilloscope driver."""

    def test_initialization(self, scope, mock_adapter):
        """Test oscilloscope initialization."""
        assert scope.adapter == mock_adapter

    def test_get_active_channels_multiple(self, scope, mock_adapter):
        """Test detecting multiple active channels."""
        mock_adapter.ask.side_effect = ["1", "1", "0", "1"]

        channels = scope.get_active_channels()

        assert mock_adapter.ask.call_count == 4
        assert channels == ["CH1", "CH2", "CH4"]

    def test_get_active_channels_single(self, scope, mock_adapter):
        """Test detecting single active channel."""
        mock_adapter.ask.side_effect = ["0", "1", "0", "0"]

        channels = scope.get_active_channels()

        assert channels == ["CH2"]

    def test_get_active_channels_none(self, scope, mock_adapter):
        """Test when no channels are active."""
        mock_adapter.ask.side_effect = ["0", "0", "0", "0"]

        channels = scope.get_active_channels()

        assert channels == []

    def test_get_active_channels_invalid_response(self, scope, mock_adapter):
        """Test handling of invalid channel responses."""
        mock_adapter.ask.side_effect = ["1", "invalid", "1", "error"]

        channels = scope.get_active_channels()

        assert channels == ["CH1", "CH3"]

    def test_set_record_length(self, scope, mock_adapter):
        """Test setting record length."""
        mock_adapter.read_line.return_value = "5000"

        actual_length = scope.set_record_length(5000)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("HORizontal:RECOrdlength 5000" in call for call in write_calls)
        assert any("HORizontal:RECOrdlength?" in call for call in write_calls)
        assert actual_length == 5000

    def test_set_record_length_large_warning(self, scope, mock_adapter):
        """Test warning for very large record length."""
        mock_adapter.read_line.return_value = "20000"

        with patch('builtins.print') as mock_print:
            actual_length = scope.set_record_length(20000)

        assert actual_length == 20000
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("WARNING" in str(call) for call in print_calls)

    def test_parse_preamble_valid(self, scope):
        """Test parsing valid preamble string."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1, DC coupling, 2.0E0 V/div, 1.0E-4 s/div, 10000 points, Sample mode";10000;Y;s;1.0E-6;0;V;1.0E-2;-50;0.0'

        preamble = scope._parse_preamble(preamble_str)

        assert preamble['description'] == 'Ch1, DC coupling, 2.0E0 V/div, 1.0E-4 s/div, 10000 points, Sample mode'
        assert preamble['nr_pt'] == 10000
        assert preamble['xunit'] == 's'
        assert preamble['xincr'] == pytest.approx(1.0e-6)
        assert preamble['pt_off'] == 0
        assert preamble['yunit'] == 'V'
        assert preamble['ymult'] == pytest.approx(1.0e-2)
        assert preamble['yoff'] == -50
        assert preamble['yzero'] == 0.0

    def test_parse_preamble_incomplete(self, scope):
        """Test error handling for incomplete preamble."""
        preamble_str = '1;0;ASC;RP;BIN;"description";10000;Y;s;1.0E-6'

        with pytest.raises(ValueError) as excinfo:
            scope._parse_preamble(preamble_str)

        assert "Incomplete preamble" in str(excinfo.value)
        assert "got 10 fields" in str(excinfo.value)

    def test_read_waveform_complete(self, scope, mock_adapter):
        """Test complete waveform reading workflow."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";8;Y;s;1.0E-6;0;V;0.01;0;0.0'
        binary_data = struct.pack('>8h', 100, 200, 300, 400, 500, 600, 700, 800)

        mock_adapter.ask.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = scope.read_waveform('CH1')

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("DATa:SOUrce CH1" in str(call) for call in write_calls)
        assert any("DATa:ENCdg RIBinary" in str(call) for call in write_calls)
        assert any("DATa:WIDth 2" in str(call) for call in write_calls)
        assert any("DATa:STARt 1" in str(call) for call in write_calls)

        assert isinstance(waveform, WaveformData)
        assert waveform.channel == 'CH1'
        assert len(waveform.voltage) == 8
        assert len(waveform.time) == 8

        expected_voltages = [100 * 0.01, 200 * 0.01, 300 * 0.01, 400 * 0.01,
                            500 * 0.01, 600 * 0.01, 700 * 0.01, 800 * 0.01]
        np.testing.assert_array_almost_equal(waveform.voltage, expected_voltages)

        expected_times = [i * 1e-6 for i in range(8)]
        np.testing.assert_array_almost_equal(waveform.time, expected_times)

    def test_read_waveform_with_offset(self, scope, mock_adapter):
        """Test waveform reading with yoff and yzero."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";4;Y;s;1.0E-6;0;V;0.02;-100;1.5'
        binary_data = struct.pack('>4h', 100, 200, 300, 400)

        mock_adapter.ask.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = scope.read_waveform('CH2')

        expected_voltage_0 = ((100 - (-100)) * 0.02) + 1.5
        assert waveform.voltage[0] == pytest.approx(expected_voltage_0)

    def test_read_waveform_incomplete_data(self, scope, mock_adapter):
        """Test error handling when incomplete binary data received."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";10;Y;s;1.0E-6;0;V;0.01;0;0.0'
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)

        mock_adapter.ask.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        with pytest.raises(ValueError) as excinfo:
            scope.read_waveform('CH1')

        assert "Incomplete data" in str(excinfo.value)

    def test_check_errors_no_error(self, scope, mock_adapter):
        """Test error checking with no errors."""
        mock_adapter.ask.return_value = '0,"No events to report - queue empty"'

        result = scope.check_errors()

        assert result == '0,"No events to report - queue empty"'
        mock_adapter.ask.assert_called_with("ALLEV?")

    def test_check_errors_with_error(self, scope, mock_adapter):
        """Test error checking when error exists."""
        mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = scope.check_errors()

        assert result == '-113,"Undefined header"'
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error" in str(call) for call in print_calls)

    def test_waveform_data_dataclass(self):
        """Test WaveformData dataclass structure."""
        time_array = np.array([0.0, 1e-6, 2e-6])
        voltage_array = np.array([0.0, 1.0, 2.0])
        preamble = {'nr_pt': 3, 'xincr': 1e-6}

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

    def test_stop_acquisition(self, scope, mock_adapter):
        """Test stopping acquisition (freezing waveform)."""
        scope.stop_acquisition()

        mock_adapter.write.assert_called_once_with("ACQuire:STATE STOP")

    def test_run_acquisition(self, scope, mock_adapter):
        """Test resuming acquisition."""
        scope.run_acquisition()

        mock_adapter.write.assert_called_once_with("ACQuire:STATE RUN")


class TestTDS460AMisc:
    """Tests for miscellaneous TDS460A commands (ID, autoset)."""

    def test_get_id(self, scope, mock_adapter):
        """Test querying instrument identification."""
        mock_adapter.ask.return_value = "TEKTRONIX,TDS 460A,0,CF:91.1CT FV:v1.0"

        result = scope.get_id()

        mock_adapter.ask.assert_called_with("*IDN?")
        assert result == "TEKTRONIX,TDS 460A,0,CF:91.1CT FV:v1.0"
        assert isinstance(result, str)

    def test_get_id_empty_response(self, scope, mock_adapter):
        """Test get_id with empty string response."""
        mock_adapter.ask.return_value = ""

        result = scope.get_id()

        mock_adapter.ask.assert_called_with("*IDN?")
        assert result == ""

    def test_autoset(self, scope, mock_adapter):
        """Test autoset sends correct SCPI command."""
        scope.autoset()

        mock_adapter.write.assert_called_once_with("AUTOSet EXECUTE")


class TestTDS460AAcquisition:
    """Tests for acquisition control methods."""

    def test_single_acquisition(self, scope, mock_adapter):
        """Test single acquisition sends stop-after and run commands."""
        scope.single_acquisition()

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "ACQuire:STOPAfter SEQuence"
        assert calls[1].args[0] == "ACQuire:STATE RUN"

    def test_get_acquisition_state_running(self, scope, mock_adapter):
        """Test querying acquisition state when running."""
        mock_adapter.ask.return_value = "1"

        result = scope.get_acquisition_state()

        mock_adapter.ask.assert_called_with("ACQuire:STATE?")
        assert result == "1"
        assert isinstance(result, str)

    def test_get_acquisition_state_stopped(self, scope, mock_adapter):
        """Test querying acquisition state when stopped."""
        mock_adapter.ask.return_value = "0"

        result = scope.get_acquisition_state()

        mock_adapter.ask.assert_called_with("ACQuire:STATE?")
        assert result == "0"

    def test_get_acquisition_mode(self, scope, mock_adapter):
        """Test querying acquisition mode."""
        mock_adapter.ask.return_value = "SAMple"

        result = scope.get_acquisition_mode()

        mock_adapter.ask.assert_called_with("ACQuire:MODe?")
        assert result == "SAMple"
        assert isinstance(result, str)

    def test_get_acquisition_mode_average(self, scope, mock_adapter):
        """Test querying acquisition mode when set to average."""
        mock_adapter.ask.return_value = "AVErage"

        result = scope.get_acquisition_mode()

        assert result == "AVErage"

    def test_set_acquisition_mode_sample(self, scope, mock_adapter):
        """Test setting acquisition mode to sample."""
        scope.set_acquisition_mode("SAMple")

        mock_adapter.write.assert_called_once_with("ACQuire:MODe SAMple")

    def test_set_acquisition_mode_average(self, scope, mock_adapter):
        """Test setting acquisition mode to average."""
        scope.set_acquisition_mode("AVErage")

        mock_adapter.write.assert_called_once_with("ACQuire:MODe AVErage")

    def test_set_acquisition_mode_peakdetect(self, scope, mock_adapter):
        """Test setting acquisition mode to peak detect."""
        scope.set_acquisition_mode("PEAKdetect")

        mock_adapter.write.assert_called_once_with("ACQuire:MODe PEAKdetect")

    def test_set_acquisition_mode_hires(self, scope, mock_adapter):
        """Test setting acquisition mode to high resolution."""
        scope.set_acquisition_mode("HIRes")

        mock_adapter.write.assert_called_once_with("ACQuire:MODe HIRes")

    def test_set_acquisition_mode_envelope(self, scope, mock_adapter):
        """Test setting acquisition mode to envelope."""
        scope.set_acquisition_mode("ENVelope")

        mock_adapter.write.assert_called_once_with("ACQuire:MODe ENVelope")

    def test_get_num_averages(self, scope, mock_adapter):
        """Test querying number of averages."""
        mock_adapter.ask.return_value = "64"

        result = scope.get_num_averages()

        mock_adapter.ask.assert_called_with("ACQuire:NUMAVg?")
        assert result == 64
        assert isinstance(result, int)

    def test_get_num_averages_large(self, scope, mock_adapter):
        """Test querying a large number of averages."""
        mock_adapter.ask.return_value = "10000"

        result = scope.get_num_averages()

        assert result == 10000

    def test_set_num_averages(self, scope, mock_adapter):
        """Test setting number of averages."""
        scope.set_num_averages(128)

        mock_adapter.write.assert_called_once_with("ACQuire:NUMAVg 128")

    def test_set_num_averages_small(self, scope, mock_adapter):
        """Test setting a small number of averages."""
        scope.set_num_averages(2)

        mock_adapter.write.assert_called_once_with("ACQuire:NUMAVg 2")


class TestTDS460AHorizontal:
    """Tests for horizontal (timebase) control methods."""

    def test_get_horizontal_scale(self, scope, mock_adapter):
        """Test querying horizontal time/div."""
        mock_adapter.ask.return_value = "1.0E-4"

        result = scope.get_horizontal_scale()

        mock_adapter.ask.assert_called_with("HORizontal:MAIn:SCAle?")
        assert result == pytest.approx(1.0e-4)
        assert isinstance(result, float)

    def test_get_horizontal_scale_fast(self, scope, mock_adapter):
        """Test querying horizontal scale at fast timebase."""
        mock_adapter.ask.return_value = "2.0E-9"

        result = scope.get_horizontal_scale()

        assert result == pytest.approx(2.0e-9)

    def test_get_horizontal_scale_slow(self, scope, mock_adapter):
        """Test querying horizontal scale at slow timebase."""
        mock_adapter.ask.return_value = "10.0"

        result = scope.get_horizontal_scale()

        assert result == pytest.approx(10.0)

    def test_set_horizontal_scale(self, scope, mock_adapter):
        """Test setting horizontal time/div."""
        scope.set_horizontal_scale(5.0e-6)

        mock_adapter.write.assert_called_once_with("HORizontal:MAIn:SCAle 5e-06")

    def test_set_horizontal_scale_milliseconds(self, scope, mock_adapter):
        """Test setting horizontal scale to milliseconds range."""
        scope.set_horizontal_scale(0.001)

        mock_adapter.write.assert_called_once_with("HORizontal:MAIn:SCAle 0.001")

    def test_get_record_length(self, scope, mock_adapter):
        """Test querying record length via ask (simple getter)."""
        mock_adapter.ask.return_value = "10000"

        result = scope.get_record_length()

        mock_adapter.ask.assert_called_with("HORizontal:RECOrdlength?")
        assert result == 10000
        assert isinstance(result, int)

    def test_get_record_length_small(self, scope, mock_adapter):
        """Test querying small record length."""
        mock_adapter.ask.return_value = "500"

        result = scope.get_record_length()

        assert result == 500

    def test_get_horizontal_position(self, scope, mock_adapter):
        """Test querying horizontal trigger position."""
        mock_adapter.ask.return_value = "50.0"

        result = scope.get_horizontal_position()

        mock_adapter.ask.assert_called_with("HORizontal:POSition?")
        assert result == pytest.approx(50.0)
        assert isinstance(result, float)

    def test_get_horizontal_position_left(self, scope, mock_adapter):
        """Test querying horizontal position at left edge."""
        mock_adapter.ask.return_value = "0.0"

        result = scope.get_horizontal_position()

        assert result == pytest.approx(0.0)

    def test_get_horizontal_position_right(self, scope, mock_adapter):
        """Test querying horizontal position at right edge."""
        mock_adapter.ask.return_value = "100.0"

        result = scope.get_horizontal_position()

        assert result == pytest.approx(100.0)

    def test_set_horizontal_position(self, scope, mock_adapter):
        """Test setting horizontal trigger position."""
        scope.set_horizontal_position(25.0)

        mock_adapter.write.assert_called_once_with("HORizontal:POSition 25.0")

    def test_set_horizontal_position_center(self, scope, mock_adapter):
        """Test setting horizontal position to center."""
        scope.set_horizontal_position(50.0)

        mock_adapter.write.assert_called_once_with("HORizontal:POSition 50.0")


class TestTDS460AVertical:
    """Tests for vertical (per-channel) control methods."""

    def test_get_channel_scale(self, scope, mock_adapter):
        """Test querying channel volts/div."""
        mock_adapter.ask.return_value = "2.0"

        result = scope.get_channel_scale("CH1")

        mock_adapter.ask.assert_called_with("CH1:SCAle?")
        assert result == pytest.approx(2.0)
        assert isinstance(result, float)

    def test_get_channel_scale_millivolts(self, scope, mock_adapter):
        """Test querying channel scale in millivolt range."""
        mock_adapter.ask.return_value = "5.0E-3"

        result = scope.get_channel_scale("CH2")

        mock_adapter.ask.assert_called_with("CH2:SCAle?")
        assert result == pytest.approx(5.0e-3)

    def test_get_channel_scale_all_channels(self, scope, mock_adapter):
        """Test querying scale for each channel."""
        for ch_num in range(1, 5):
            channel = f"CH{ch_num}"
            mock_adapter.ask.return_value = "1.0"

            result = scope.get_channel_scale(channel)

            mock_adapter.ask.assert_called_with(f"{channel}:SCAle?")
            assert result == pytest.approx(1.0)

    def test_set_channel_scale(self, scope, mock_adapter):
        """Test setting channel volts/div."""
        scope.set_channel_scale("CH1", 0.5)

        mock_adapter.write.assert_called_once_with("CH1:SCAle 0.5")

    def test_set_channel_scale_millivolts(self, scope, mock_adapter):
        """Test setting channel scale to millivolt range."""
        scope.set_channel_scale("CH3", 0.002)

        mock_adapter.write.assert_called_once_with("CH3:SCAle 0.002")

    def test_get_channel_offset(self, scope, mock_adapter):
        """Test querying channel vertical offset."""
        mock_adapter.ask.return_value = "1.5"

        result = scope.get_channel_offset("CH1")

        mock_adapter.ask.assert_called_with("CH1:OFFSet?")
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_get_channel_offset_negative(self, scope, mock_adapter):
        """Test querying negative channel offset."""
        mock_adapter.ask.return_value = "-3.2"

        result = scope.get_channel_offset("CH2")

        mock_adapter.ask.assert_called_with("CH2:OFFSet?")
        assert result == pytest.approx(-3.2)

    def test_get_channel_offset_zero(self, scope, mock_adapter):
        """Test querying zero channel offset."""
        mock_adapter.ask.return_value = "0.0"

        result = scope.get_channel_offset("CH4")

        assert result == pytest.approx(0.0)

    def test_set_channel_offset(self, scope, mock_adapter):
        """Test setting channel vertical offset."""
        scope.set_channel_offset("CH1", 2.5)

        mock_adapter.write.assert_called_once_with("CH1:OFFSet 2.5")

    def test_set_channel_offset_negative(self, scope, mock_adapter):
        """Test setting negative channel offset."""
        scope.set_channel_offset("CH3", -1.0)

        mock_adapter.write.assert_called_once_with("CH3:OFFSet -1.0")

    def test_get_channel_coupling(self, scope, mock_adapter):
        """Test querying channel coupling."""
        mock_adapter.ask.return_value = "DC"

        result = scope.get_channel_coupling("CH1")

        mock_adapter.ask.assert_called_with("CH1:COUPling?")
        assert result == "DC"
        assert isinstance(result, str)

    def test_get_channel_coupling_ac(self, scope, mock_adapter):
        """Test querying AC channel coupling."""
        mock_adapter.ask.return_value = "AC"

        result = scope.get_channel_coupling("CH2")

        mock_adapter.ask.assert_called_with("CH2:COUPling?")
        assert result == "AC"

    def test_get_channel_coupling_gnd(self, scope, mock_adapter):
        """Test querying GND channel coupling."""
        mock_adapter.ask.return_value = "GND"

        result = scope.get_channel_coupling("CH1")

        assert result == "GND"

    def test_set_channel_coupling_dc(self, scope, mock_adapter):
        """Test setting channel coupling to DC."""
        scope.set_channel_coupling("CH1", "DC")

        mock_adapter.write.assert_called_once_with("CH1:COUPling DC")

    def test_set_channel_coupling_ac(self, scope, mock_adapter):
        """Test setting channel coupling to AC."""
        scope.set_channel_coupling("CH2", "AC")

        mock_adapter.write.assert_called_once_with("CH2:COUPling AC")

    def test_set_channel_coupling_gnd(self, scope, mock_adapter):
        """Test setting channel coupling to GND."""
        scope.set_channel_coupling("CH4", "GND")

        mock_adapter.write.assert_called_once_with("CH4:COUPling GND")

    def test_get_channel_bandwidth(self, scope, mock_adapter):
        """Test querying channel bandwidth limit."""
        mock_adapter.ask.return_value = "FULl"

        result = scope.get_channel_bandwidth("CH1")

        mock_adapter.ask.assert_called_with("CH1:BANdwidth?")
        assert result == "FULl"
        assert isinstance(result, str)

    def test_get_channel_bandwidth_twenty(self, scope, mock_adapter):
        """Test querying 20MHz bandwidth limit."""
        mock_adapter.ask.return_value = "TWEnty"

        result = scope.get_channel_bandwidth("CH3")

        mock_adapter.ask.assert_called_with("CH3:BANdwidth?")
        assert result == "TWEnty"

    def test_set_channel_bandwidth_full(self, scope, mock_adapter):
        """Test setting channel bandwidth to full."""
        scope.set_channel_bandwidth("CH1", "FULl")

        mock_adapter.write.assert_called_once_with("CH1:BANdwidth FULl")

    def test_set_channel_bandwidth_twenty(self, scope, mock_adapter):
        """Test setting channel bandwidth to 20MHz."""
        scope.set_channel_bandwidth("CH2", "TWEnty")

        mock_adapter.write.assert_called_once_with("CH2:BANdwidth TWEnty")

    def test_set_channel_bandwidth_onehundred(self, scope, mock_adapter):
        """Test setting channel bandwidth to 100MHz."""
        scope.set_channel_bandwidth("CH4", "ONEhundred")

        mock_adapter.write.assert_called_once_with("CH4:BANdwidth ONEhundred")

    def test_set_channel_display_on(self, scope, mock_adapter):
        """Test turning channel display on."""
        scope.set_channel_display("CH1", True)

        mock_adapter.write.assert_called_once_with("SELect:CH1 ON")

    def test_set_channel_display_off(self, scope, mock_adapter):
        """Test turning channel display off."""
        scope.set_channel_display("CH2", False)

        mock_adapter.write.assert_called_once_with("SELect:CH2 OFF")

    def test_set_channel_display_on_ch4(self, scope, mock_adapter):
        """Test turning CH4 display on."""
        scope.set_channel_display("CH4", True)

        mock_adapter.write.assert_called_once_with("SELect:CH4 ON")

    def test_set_channel_display_off_ch3(self, scope, mock_adapter):
        """Test turning CH3 display off."""
        scope.set_channel_display("CH3", False)

        mock_adapter.write.assert_called_once_with("SELect:CH3 OFF")


class TestTDS460ATrigger:
    """Tests for trigger control methods."""

    def test_get_trigger_source(self, scope, mock_adapter):
        """Test querying trigger source."""
        mock_adapter.ask.return_value = "CH1"

        result = scope.get_trigger_source()

        mock_adapter.ask.assert_called_with("TRIGger:MAIn:EDGE:SOUrce?")
        assert result == "CH1"
        assert isinstance(result, str)

    def test_get_trigger_source_ext(self, scope, mock_adapter):
        """Test querying external trigger source."""
        mock_adapter.ask.return_value = "EXT"

        result = scope.get_trigger_source()

        assert result == "EXT"

    def test_get_trigger_source_line(self, scope, mock_adapter):
        """Test querying line trigger source."""
        mock_adapter.ask.return_value = "LINE"

        result = scope.get_trigger_source()

        assert result == "LINE"

    def test_set_trigger_source_ch1(self, scope, mock_adapter):
        """Test setting trigger source to CH1."""
        scope.set_trigger_source("CH1")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SOUrce CH1")

    def test_set_trigger_source_ch3(self, scope, mock_adapter):
        """Test setting trigger source to CH3."""
        scope.set_trigger_source("CH3")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SOUrce CH3")

    def test_set_trigger_source_ext(self, scope, mock_adapter):
        """Test setting trigger source to external."""
        scope.set_trigger_source("EXT")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SOUrce EXT")

    def test_set_trigger_source_line(self, scope, mock_adapter):
        """Test setting trigger source to line."""
        scope.set_trigger_source("LINE")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SOUrce LINE")

    def test_get_trigger_level(self, scope, mock_adapter):
        """Test querying trigger level."""
        mock_adapter.ask.return_value = "1.5"

        result = scope.get_trigger_level()

        mock_adapter.ask.assert_called_with("TRIGger:MAIn:LEVel?")
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_get_trigger_level_negative(self, scope, mock_adapter):
        """Test querying negative trigger level."""
        mock_adapter.ask.return_value = "-0.75"

        result = scope.get_trigger_level()

        assert result == pytest.approx(-0.75)

    def test_get_trigger_level_zero(self, scope, mock_adapter):
        """Test querying zero trigger level."""
        mock_adapter.ask.return_value = "0.0"

        result = scope.get_trigger_level()

        assert result == pytest.approx(0.0)

    def test_get_trigger_level_scientific_notation(self, scope, mock_adapter):
        """Test querying trigger level in scientific notation."""
        mock_adapter.ask.return_value = "2.5E-1"

        result = scope.get_trigger_level()

        assert result == pytest.approx(0.25)

    def test_set_trigger_level(self, scope, mock_adapter):
        """Test setting trigger level."""
        scope.set_trigger_level(2.0)

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:LEVel 2.0")

    def test_set_trigger_level_negative(self, scope, mock_adapter):
        """Test setting negative trigger level."""
        scope.set_trigger_level(-1.5)

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:LEVel -1.5")

    def test_set_trigger_level_zero(self, scope, mock_adapter):
        """Test setting trigger level to zero."""
        scope.set_trigger_level(0.0)

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:LEVel 0.0")

    def test_get_trigger_slope(self, scope, mock_adapter):
        """Test querying trigger slope."""
        mock_adapter.ask.return_value = "RISe"

        result = scope.get_trigger_slope()

        mock_adapter.ask.assert_called_with("TRIGger:MAIn:EDGE:SLOpe?")
        assert result == "RISe"
        assert isinstance(result, str)

    def test_get_trigger_slope_fall(self, scope, mock_adapter):
        """Test querying falling trigger slope."""
        mock_adapter.ask.return_value = "FALL"

        result = scope.get_trigger_slope()

        assert result == "FALL"

    def test_set_trigger_slope_rise(self, scope, mock_adapter):
        """Test setting trigger slope to rising."""
        scope.set_trigger_slope("RISe")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SLOpe RISe")

    def test_set_trigger_slope_fall(self, scope, mock_adapter):
        """Test setting trigger slope to falling."""
        scope.set_trigger_slope("FALL")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:EDGE:SLOpe FALL")

    def test_get_trigger_mode(self, scope, mock_adapter):
        """Test querying trigger mode."""
        mock_adapter.ask.return_value = "AUTO"

        result = scope.get_trigger_mode()

        mock_adapter.ask.assert_called_with("TRIGger:MAIn:MODe?")
        assert result == "AUTO"
        assert isinstance(result, str)

    def test_get_trigger_mode_normal(self, scope, mock_adapter):
        """Test querying normal trigger mode."""
        mock_adapter.ask.return_value = "NORMal"

        result = scope.get_trigger_mode()

        assert result == "NORMal"

    def test_set_trigger_mode_auto(self, scope, mock_adapter):
        """Test setting trigger mode to auto."""
        scope.set_trigger_mode("AUTO")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:MODe AUTO")

    def test_set_trigger_mode_normal(self, scope, mock_adapter):
        """Test setting trigger mode to normal."""
        scope.set_trigger_mode("NORMal")

        mock_adapter.write.assert_called_once_with("TRIGger:MAIn:MODe NORMal")


class TestTDS460AMeasurements:
    """Tests for measurement methods (immediate and slot-based)."""

    def test_measure_immediate(self, scope, mock_adapter):
        """Test taking an immediate measurement."""
        mock_adapter.ask.return_value = "3.14159"

        result = scope.measure_immediate("CH1", "FREQuency")

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "MEASUrement:IMMed:SOUrce1 CH1"
        assert calls[1].args[0] == "MEASUrement:IMMed:TYPe FREQuency"
        mock_adapter.ask.assert_called_with("MEASUrement:IMMed:VALue?")
        assert result == pytest.approx(3.14159)
        assert isinstance(result, float)

    def test_measure_immediate_pk2pk(self, scope, mock_adapter):
        """Test immediate peak-to-peak measurement."""
        mock_adapter.ask.return_value = "5.6"

        result = scope.measure_immediate("CH2", "PK2pk")

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "MEASUrement:IMMed:SOUrce1 CH2"
        assert calls[1].args[0] == "MEASUrement:IMMed:TYPe PK2pk"
        assert result == pytest.approx(5.6)

    def test_measure_immediate_mean(self, scope, mock_adapter):
        """Test immediate mean measurement."""
        mock_adapter.ask.return_value = "1.65"

        result = scope.measure_immediate("CH3", "MEAN")

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "MEASUrement:IMMed:SOUrce1 CH3"
        assert calls[1].args[0] == "MEASUrement:IMMed:TYPe MEAN"
        assert result == pytest.approx(1.65)

    def test_measure_immediate_rise_time(self, scope, mock_adapter):
        """Test immediate rise time measurement."""
        mock_adapter.ask.return_value = "2.5E-9"

        result = scope.measure_immediate("CH1", "RISe")

        assert result == pytest.approx(2.5e-9)

    def test_measure_immediate_minimum(self, scope, mock_adapter):
        """Test immediate minimum measurement with negative value."""
        mock_adapter.ask.return_value = "-0.125"

        result = scope.measure_immediate("CH4", "MINImum")

        assert result == pytest.approx(-0.125)

    def test_configure_measurement_slot(self, scope, mock_adapter):
        """Test configuring a measurement slot."""
        scope.configure_measurement_slot(1, "CH1", "FREQuency")

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 3
        assert calls[0].args[0] == "MEASUrement:MEAS1:SOUrce CH1"
        assert calls[1].args[0] == "MEASUrement:MEAS1:TYPe FREQuency"
        assert calls[2].args[0] == "MEASUrement:MEAS1:STATE ON"

    def test_configure_measurement_slot_2(self, scope, mock_adapter):
        """Test configuring measurement slot 2."""
        scope.configure_measurement_slot(2, "CH3", "PK2pk")

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 3
        assert calls[0].args[0] == "MEASUrement:MEAS2:SOUrce CH3"
        assert calls[1].args[0] == "MEASUrement:MEAS2:TYPe PK2pk"
        assert calls[2].args[0] == "MEASUrement:MEAS2:STATE ON"

    def test_configure_measurement_slot_3(self, scope, mock_adapter):
        """Test configuring measurement slot 3."""
        scope.configure_measurement_slot(3, "CH2", "MEAN")

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "MEASUrement:MEAS3:SOUrce CH2"
        assert calls[1].args[0] == "MEASUrement:MEAS3:TYPe MEAN"
        assert calls[2].args[0] == "MEASUrement:MEAS3:STATE ON"

    def test_configure_measurement_slot_4(self, scope, mock_adapter):
        """Test configuring measurement slot 4."""
        scope.configure_measurement_slot(4, "CH4", "PERIod")

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "MEASUrement:MEAS4:SOUrce CH4"
        assert calls[1].args[0] == "MEASUrement:MEAS4:TYPe PERIod"
        assert calls[2].args[0] == "MEASUrement:MEAS4:STATE ON"

    def test_read_measurement_slot(self, scope, mock_adapter):
        """Test reading a measurement slot value."""
        mock_adapter.ask.return_value = "1000000.0"

        result = scope.read_measurement_slot(1)

        mock_adapter.ask.assert_called_with("MEASUrement:MEAS1:VALue?")
        assert result == pytest.approx(1000000.0)
        assert isinstance(result, float)

    def test_read_measurement_slot_2(self, scope, mock_adapter):
        """Test reading measurement slot 2."""
        mock_adapter.ask.return_value = "3.3"

        result = scope.read_measurement_slot(2)

        mock_adapter.ask.assert_called_with("MEASUrement:MEAS2:VALue?")
        assert result == pytest.approx(3.3)

    def test_read_measurement_slot_3(self, scope, mock_adapter):
        """Test reading measurement slot 3."""
        mock_adapter.ask.return_value = "2.5E-6"

        result = scope.read_measurement_slot(3)

        mock_adapter.ask.assert_called_with("MEASUrement:MEAS3:VALue?")
        assert result == pytest.approx(2.5e-6)

    def test_read_measurement_slot_4(self, scope, mock_adapter):
        """Test reading measurement slot 4."""
        mock_adapter.ask.return_value = "-0.05"

        result = scope.read_measurement_slot(4)

        mock_adapter.ask.assert_called_with("MEASUrement:MEAS4:VALue?")
        assert result == pytest.approx(-0.05)


class TestTDS460ACursors:
    """Tests for cursor control methods."""

    def test_get_cursor_function(self, scope, mock_adapter):
        """Test querying cursor function."""
        mock_adapter.ask.return_value = "OFF"

        result = scope.get_cursor_function()

        mock_adapter.ask.assert_called_with("CURSor:FUNCtion?")
        assert result == "OFF"
        assert isinstance(result, str)

    def test_get_cursor_function_hbars(self, scope, mock_adapter):
        """Test querying cursor function when set to HBars."""
        mock_adapter.ask.return_value = "HBArs"

        result = scope.get_cursor_function()

        assert result == "HBArs"

    def test_set_cursor_function_hbars(self, scope, mock_adapter):
        """Test setting cursor function to horizontal bars."""
        scope.set_cursor_function("HBArs")

        mock_adapter.write.assert_called_once_with("CURSor:FUNCtion HBArs")

    def test_set_cursor_function_vbars(self, scope, mock_adapter):
        """Test setting cursor function to vertical bars."""
        scope.set_cursor_function("VBArs")

        mock_adapter.write.assert_called_once_with("CURSor:FUNCtion VBArs")

    def test_set_cursor_function_paired(self, scope, mock_adapter):
        """Test setting cursor function to paired."""
        scope.set_cursor_function("PAIred")

        mock_adapter.write.assert_called_once_with("CURSor:FUNCtion PAIred")

    def test_set_cursor_function_off(self, scope, mock_adapter):
        """Test turning cursors off."""
        scope.set_cursor_function("OFF")

        mock_adapter.write.assert_called_once_with("CURSor:FUNCtion OFF")

    def test_set_hbar_positions(self, scope, mock_adapter):
        """Test setting horizontal bar cursor positions."""
        scope.set_hbar_positions(1.0, 3.0)

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "CURSor:HBArs:POSITION1 1.0"
        assert calls[1].args[0] == "CURSor:HBArs:POSITION2 3.0"

    def test_set_hbar_positions_negative(self, scope, mock_adapter):
        """Test setting horizontal bar cursors with negative values."""
        scope.set_hbar_positions(-2.5, 2.5)

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "CURSor:HBArs:POSITION1 -2.5"
        assert calls[1].args[0] == "CURSor:HBArs:POSITION2 2.5"

    def test_set_hbar_positions_zero(self, scope, mock_adapter):
        """Test setting horizontal bar cursors at zero."""
        scope.set_hbar_positions(0.0, 0.0)

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "CURSor:HBArs:POSITION1 0.0"
        assert calls[1].args[0] == "CURSor:HBArs:POSITION2 0.0"

    def test_get_hbar_delta(self, scope, mock_adapter):
        """Test querying horizontal bar delta."""
        mock_adapter.ask.return_value = "2.0"

        result = scope.get_hbar_delta()

        mock_adapter.ask.assert_called_with("CURSor:HBArs:DELTa?")
        assert result == pytest.approx(2.0)
        assert isinstance(result, float)

    def test_get_hbar_delta_negative(self, scope, mock_adapter):
        """Test querying negative horizontal bar delta."""
        mock_adapter.ask.return_value = "-1.5"

        result = scope.get_hbar_delta()

        assert result == pytest.approx(-1.5)

    def test_get_hbar_delta_small(self, scope, mock_adapter):
        """Test querying small horizontal bar delta."""
        mock_adapter.ask.return_value = "5.0E-3"

        result = scope.get_hbar_delta()

        assert result == pytest.approx(5.0e-3)

    def test_set_vbar_positions(self, scope, mock_adapter):
        """Test setting vertical bar cursor positions."""
        scope.set_vbar_positions(1.0e-6, 5.0e-6)

        calls = mock_adapter.write.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "CURSor:VBArs:POSITION1 1e-06"
        assert calls[1].args[0] == "CURSor:VBArs:POSITION2 5e-06"

    def test_set_vbar_positions_negative(self, scope, mock_adapter):
        """Test setting vertical bar cursors with negative time values."""
        scope.set_vbar_positions(-1.0e-3, 1.0e-3)

        calls = mock_adapter.write.call_args_list
        assert calls[0].args[0] == "CURSor:VBArs:POSITION1 -0.001"
        assert calls[1].args[0] == "CURSor:VBArs:POSITION2 0.001"

    def test_get_vbar_delta(self, scope, mock_adapter):
        """Test querying vertical bar delta."""
        mock_adapter.ask.return_value = "4.0E-6"

        result = scope.get_vbar_delta()

        mock_adapter.ask.assert_called_with("CURSor:VBArs:DELTa?")
        assert result == pytest.approx(4.0e-6)
        assert isinstance(result, float)

    def test_get_vbar_delta_negative(self, scope, mock_adapter):
        """Test querying negative vertical bar delta."""
        mock_adapter.ask.return_value = "-2.0E-3"

        result = scope.get_vbar_delta()

        assert result == pytest.approx(-2.0e-3)

    def test_get_vbar_delta_large(self, scope, mock_adapter):
        """Test querying large vertical bar delta."""
        mock_adapter.ask.return_value = "1.0"

        result = scope.get_vbar_delta()

        assert result == pytest.approx(1.0)
