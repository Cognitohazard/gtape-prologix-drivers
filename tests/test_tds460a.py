"""Unit tests for TDS460A oscilloscope driver."""

import pytest
from unittest.mock import Mock, patch
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
        mock_adapter.ask.return_value = "5000"

        actual_length = scope.set_record_length(5000)

        mock_adapter.write.assert_called_with("HORizontal:RECOrdlength 5000")
        mock_adapter.ask.assert_called_with("HORizontal:RECOrdlength?")
        assert actual_length == 5000

    def test_set_record_length_large_warning(self, scope, mock_adapter):
        """Test warning for very large record length."""
        mock_adapter.ask.return_value = "20000"

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

        mock_adapter.ask.return_value = "8"
        mock_adapter.read_line.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = scope.read_waveform('CH1')

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("DATa:SOUrce CH1" in str(call) for call in write_calls)
        assert any("DATa:ENCdg RIBinary" in str(call) for call in write_calls)
        assert any("DATa:WIDth 2" in str(call) for call in write_calls)

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

        mock_adapter.ask.return_value = "4"
        mock_adapter.read_line.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = scope.read_waveform('CH2')

        expected_voltage_0 = ((100 - (-100)) * 0.02) + 1.5
        assert waveform.voltage[0] == pytest.approx(expected_voltage_0)

    def test_read_waveform_incomplete_data(self, scope, mock_adapter):
        """Test error handling when incomplete binary data received."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";10;Y;s;1.0E-6;0;V;0.01;0;0.0'
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)

        mock_adapter.ask.return_value = "10"
        mock_adapter.read_line.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        with pytest.raises(ValueError) as excinfo:
            scope.read_waveform('CH1')

        assert "Incomplete data" in str(excinfo.value)

    def test_read_waveform_with_record_length_param(self, scope, mock_adapter):
        """Test reading waveform with explicit record length parameter."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";5;Y;s;1.0E-6;0;V;0.01;0;0.0'
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)

        mock_adapter.read_line.return_value = preamble_str
        mock_adapter.read_binary.return_value = binary_data

        waveform = scope.read_waveform('CH3', record_length=5)

        mock_adapter.ask.assert_not_called()
        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("DATa:STOP 5" in str(call) for call in write_calls)

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
