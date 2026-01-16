"""Unit tests for TDS460A oscilloscope driver."""

import unittest
from unittest.mock import Mock, MagicMock, call, patch
import numpy as np
import struct
from gtape_prologix_drivers.instruments.tds460a import TDS460A, WaveformData


class TestTDS460A(unittest.TestCase):
    """Test cases for TDS460A oscilloscope driver."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.scope = TDS460A(self.mock_adapter)

    def test_initialization(self):
        """Test oscilloscope initialization."""
        self.assertEqual(self.scope.adapter, self.mock_adapter)

    def test_get_active_channels_multiple(self):
        """Test detecting multiple active channels."""
        # Mock responses for CH1-CH4 queries
        self.mock_adapter.ask.side_effect = ["1", "1", "0", "1"]

        channels = self.scope.get_active_channels()

        # Should have queried all 4 channels
        self.assertEqual(self.mock_adapter.ask.call_count, 4)

        # CH1, CH2, CH4 are active (not CH3)
        self.assertEqual(channels, ["CH1", "CH2", "CH4"])

    def test_get_active_channels_single(self):
        """Test detecting single active channel."""
        self.mock_adapter.ask.side_effect = ["0", "1", "0", "0"]

        channels = self.scope.get_active_channels()

        self.assertEqual(channels, ["CH2"])

    def test_get_active_channels_none(self):
        """Test when no channels are active."""
        self.mock_adapter.ask.side_effect = ["0", "0", "0", "0"]

        channels = self.scope.get_active_channels()

        self.assertEqual(channels, [])

    def test_get_active_channels_invalid_response(self):
        """Test handling of invalid channel responses."""
        # Mix of valid and invalid responses
        self.mock_adapter.ask.side_effect = ["1", "invalid", "1", "error"]

        channels = self.scope.get_active_channels()

        # Should skip invalid responses and only include valid ones
        self.assertEqual(channels, ["CH1", "CH3"])

    def test_set_record_length(self):
        """Test setting record length."""
        self.mock_adapter.ask.return_value = "5000"

        actual_length = self.scope.set_record_length(5000)

        # Verify command was sent
        self.mock_adapter.write.assert_called_with("HORizontal:RECOrdlength 5000")

        # Verify query was sent
        self.mock_adapter.ask.assert_called_with("HORizontal:RECOrdlength?")

        self.assertEqual(actual_length, 5000)

    def test_set_record_length_large_warning(self):
        """Test warning for very large record length."""
        self.mock_adapter.ask.return_value = "20000"

        with patch('builtins.print') as mock_print:
            actual_length = self.scope.set_record_length(20000)

        # Should still return the value
        self.assertEqual(actual_length, 20000)

        # Should have printed a warning (check that print was called with WARNING)
        print_calls = [str(call) for call in mock_print.call_args_list]
        warning_found = any("WARNING" in str(call) for call in print_calls)
        self.assertTrue(warning_found)

    def test_parse_preamble_valid(self):
        """Test parsing valid preamble string."""
        # Typical preamble format (semicolon-separated)
        preamble_str = '1;0;ASC;RP;BIN;"Ch1, DC coupling, 2.0E0 V/div, 1.0E-4 s/div, 10000 points, Sample mode";10000;Y;s;1.0E-6;0;V;1.0E-2;-50;0.0'

        preamble = self.scope._parse_preamble(preamble_str)

        self.assertEqual(preamble['description'], 'Ch1, DC coupling, 2.0E0 V/div, 1.0E-4 s/div, 10000 points, Sample mode')
        self.assertEqual(preamble['nr_pt'], 10000)
        self.assertEqual(preamble['xunit'], 's')
        self.assertAlmostEqual(preamble['xincr'], 1.0e-6)
        self.assertEqual(preamble['pt_off'], 0)
        self.assertEqual(preamble['yunit'], 'V')
        self.assertAlmostEqual(preamble['ymult'], 1.0e-2)
        self.assertEqual(preamble['yoff'], -50)
        self.assertEqual(preamble['yzero'], 0.0)

    def test_parse_preamble_incomplete(self):
        """Test error handling for incomplete preamble."""
        # Only 10 fields instead of 15
        preamble_str = '1;0;ASC;RP;BIN;"description";10000;Y;s;1.0E-6'

        with self.assertRaises(ValueError) as cm:
            self.scope._parse_preamble(preamble_str)

        self.assertIn("Incomplete preamble", str(cm.exception))
        self.assertIn("got 10 fields", str(cm.exception))

    def test_read_waveform_complete(self):
        """Test complete waveform reading workflow."""
        # Setup mock preamble response
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";8;Y;s;1.0E-6;0;V;0.01;0;0.0'

        # Setup mock binary data (8 points, 16-bit big-endian signed integers)
        # Values: [100, 200, 300, 400, 500, 600, 700, 800]
        binary_data = struct.pack('>8h', 100, 200, 300, 400, 500, 600, 700, 800)

        # Configure mocks
        self.mock_adapter.ask.return_value = "8"  # Record length query
        self.mock_adapter.ser = Mock()
        self.mock_adapter.ser.readline.return_value = preamble_str.encode()
        self.mock_adapter.read_binary.return_value = binary_data

        waveform = self.scope.read_waveform('CH1')

        # Verify data source was set
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("DATa:SOUrce CH1" in str(call) for call in write_calls))

        # Verify encoding was set
        self.assertTrue(any("DATa:ENCdg RIBinary" in str(call) for call in write_calls))

        # Verify width was set to 2 bytes
        self.assertTrue(any("DATa:WIDth 2" in str(call) for call in write_calls))

        # Check waveform data
        self.assertIsInstance(waveform, WaveformData)
        self.assertEqual(waveform.channel, 'CH1')
        self.assertEqual(len(waveform.voltage), 8)
        self.assertEqual(len(waveform.time), 8)

        # Verify voltage conversion formula: ((val - yoff) * ymult) + yzero
        # With ymult=0.01, yoff=0, yzero=0: voltage = val * 0.01
        expected_voltages = [100 * 0.01, 200 * 0.01, 300 * 0.01, 400 * 0.01,
                            500 * 0.01, 600 * 0.01, 700 * 0.01, 800 * 0.01]
        np.testing.assert_array_almost_equal(waveform.voltage, expected_voltages)

        # Verify time array: (i - pt_off) * xincr
        # With xincr=1e-6, pt_off=0: time[i] = i * 1e-6
        expected_times = [i * 1e-6 for i in range(8)]
        np.testing.assert_array_almost_equal(waveform.time, expected_times)

    def test_read_waveform_with_offset(self):
        """Test waveform reading with yoff and yzero."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";4;Y;s;1.0E-6;0;V;0.02;-100;1.5'

        # 4 points: [100, 200, 300, 400]
        binary_data = struct.pack('>4h', 100, 200, 300, 400)

        self.mock_adapter.ask.return_value = "4"
        self.mock_adapter.ser = Mock()
        self.mock_adapter.ser.readline.return_value = preamble_str.encode()
        self.mock_adapter.read_binary.return_value = binary_data

        waveform = self.scope.read_waveform('CH2')

        # Voltage formula: ((val - yoff) * ymult) + yzero
        # With ymult=0.02, yoff=-100, yzero=1.5
        # voltage[0] = ((100 - (-100)) * 0.02) + 1.5 = (200 * 0.02) + 1.5 = 4.0 + 1.5 = 5.5
        expected_voltage_0 = ((100 - (-100)) * 0.02) + 1.5
        self.assertAlmostEqual(waveform.voltage[0], expected_voltage_0)

    def test_read_waveform_incomplete_data(self):
        """Test error handling when incomplete binary data received."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";10;Y;s;1.0E-6;0;V;0.01;0;0.0'

        # Only 5 points instead of expected 10
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)

        self.mock_adapter.ask.return_value = "10"
        self.mock_adapter.ser = Mock()
        self.mock_adapter.ser.readline.return_value = preamble_str.encode()
        self.mock_adapter.read_binary.return_value = binary_data

        with self.assertRaises(ValueError) as cm:
            self.scope.read_waveform('CH1')

        self.assertIn("Incomplete data", str(cm.exception))

    def test_read_waveform_with_record_length_param(self):
        """Test reading waveform with explicit record length parameter."""
        preamble_str = '1;0;ASC;RP;BIN;"Ch1";5;Y;s;1.0E-6;0;V;0.01;0;0.0'
        binary_data = struct.pack('>5h', 100, 200, 300, 400, 500)

        self.mock_adapter.ser = Mock()
        self.mock_adapter.ser.readline.return_value = preamble_str.encode()
        self.mock_adapter.read_binary.return_value = binary_data

        # Pass record_length explicitly (should not query scope)
        waveform = self.scope.read_waveform('CH3', record_length=5)

        # Should NOT have queried record length
        self.mock_adapter.ask.assert_not_called()

        # Verify DATa:STOP was set correctly
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("DATa:STOP 5" in str(call) for call in write_calls))

    def test_check_errors_no_error(self):
        """Test error checking with no errors."""
        self.mock_adapter.ask.return_value = '0,"No events to report - queue empty"'

        result = self.scope.check_errors()

        self.assertEqual(result, '0,"No events to report - queue empty"')
        self.mock_adapter.ask.assert_called_with("ALLEV?")

    def test_check_errors_with_error(self):
        """Test error checking when error exists."""
        self.mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = self.scope.check_errors()

        self.assertEqual(result, '-113,"Undefined header"')

        # Should have printed the error
        print_calls = [str(call) for call in mock_print.call_args_list]
        error_found = any("Error" in str(call) for call in print_calls)
        self.assertTrue(error_found)

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

        self.assertEqual(waveform.channel, 'CH1')
        np.testing.assert_array_equal(waveform.time, time_array)
        np.testing.assert_array_equal(waveform.voltage, voltage_array)
        self.assertEqual(waveform.preamble['nr_pt'], 3)


if __name__ == '__main__':
    unittest.main()
