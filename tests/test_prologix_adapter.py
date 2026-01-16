"""Unit tests for PrologixAdapter class."""

import unittest
from unittest.mock import Mock, MagicMock, call, patch
import time
from gtape_prologix_drivers.adapter import PrologixAdapter, LF, CR, ESC, PLUS


class TestPrologixAdapter(unittest.TestCase):
    """Test cases for PrologixAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock serial.Serial to avoid needing actual hardware
        self.mock_serial_patcher = patch('gtape_prologix_drivers.adapter.serial.Serial')
        self.mock_serial_class = self.mock_serial_patcher.start()
        self.mock_serial = MagicMock()
        self.mock_serial_class.return_value = self.mock_serial
        self.mock_serial.is_open = True

    def tearDown(self):
        """Clean up after tests."""
        self.mock_serial_patcher.stop()

    def test_initialization(self):
        """Test adapter initialization and Prologix configuration."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        # Verify serial port was opened with correct parameters
        self.mock_serial_class.assert_called_once_with("COM4", 115200, timeout=6.0)

        # Verify Prologix configuration commands were sent
        expected_calls = [
            call("++mode 1\r\n".encode()),
            call("++addr 4\r\n".encode()),
            call("++auto 0\r\n".encode()),
            call("++eos 3\r\n".encode()),
            call("++eoi 1\r\n".encode()),
            call("++read_tmo_ms 4000\r\n".encode()),
        ]

        # Check that all configuration commands were sent
        actual_calls = self.mock_serial.write.call_args_list
        self.assertEqual(len(actual_calls), 6)
        for expected, actual in zip(expected_calls, actual_calls):
            self.assertEqual(expected, actual)

    def test_initialization_custom_timeout(self):
        """Test adapter initialization with custom timeout."""
        adapter = PrologixAdapter(port="COM3", gpib_address=22, timeout=5.0)

        self.mock_serial_class.assert_called_once_with("COM3", 115200, timeout=5.0)
        self.assertEqual(adapter.port, "COM3")
        self.assertEqual(adapter.address, 22)

    def test_switch_address(self):
        """Test switching GPIB address."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()  # Clear initialization calls

        # Switch to different address
        adapter.switch_address(10)

        self.assertEqual(adapter.address, 10)
        self.mock_serial.write.assert_called_with("++addr 10\r\n".encode())

    def test_switch_address_same(self):
        """Test switching to same address (should be no-op)."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # Switch to same address
        adapter.switch_address(4)

        # Should not send ++addr command
        self.mock_serial.write.assert_not_called()
        self.assertEqual(adapter.address, 4)

    def test_write_command(self):
        """Test writing SCPI command."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        adapter.write("*IDN?")

        self.mock_serial.write.assert_called_once_with("*IDN?\r\n".encode())

    def test_write_command_custom_delay(self):
        """Test write with custom delay."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        with patch('adapters.prologix.time.sleep') as mock_sleep:
            adapter.write("VOLT 5.0", delay=0.5)
            mock_sleep.assert_called_with(0.5)

    def test_read(self):
        """Test reading response from instrument."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # Mock instrument response (using readline which returns on newline)
        self.mock_serial.readline.return_value = b"HEWLETT-PACKARD,33120A,0,10.0\n\x00"

        response = adapter.read()

        # Verify ++read eoi was sent
        self.mock_serial.write.assert_called_with("++read eoi\r\n".encode())

        # Verify response was decoded and cleaned
        self.assertEqual(response, "HEWLETT-PACKARD,33120A,0,10.0")

    def test_read_with_null_bytes(self):
        """Test reading response with null bytes."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        # Response with null bytes at start and end (using readline)
        self.mock_serial.readline.return_value = b"\x00+5.01378400E+00\n\x00"

        response = adapter.read()

        self.assertEqual(response, "+5.01378400E+00")

    def test_ask(self):
        """Test query (write + read)."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        self.mock_serial.readline.return_value = b"+0,\"No error\"\n"

        response = adapter.ask("SYST:ERR?")

        # Verify command was written
        calls = self.mock_serial.write.call_args_list
        self.assertEqual(calls[0], call("SYST:ERR?\r\n".encode()))
        self.assertEqual(calls[1], call("++read eoi\r\n".encode()))

        self.assertEqual(response, '+0,"No error"')

    def test_write_binary_simple(self):
        """Test writing binary data with IEEE 488.2 format."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # Simple data: [0, 1, 2, 3]
        data = bytes([0, 1, 2, 3])

        adapter.write_binary("DATA:DAC VOLATILE, ", data)

        # Expected format: DATA:DAC VOLATILE, #14<data>\r\n
        # #1 = 1 digit in length
        # 4 = length (4 bytes)
        expected = b"DATA:DAC VOLATILE, #14" + data + b"\r\n"

        self.mock_serial.write.assert_called_once_with(expected)

    def test_write_binary_with_escaping(self):
        """Test binary data with special characters requiring escaping."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # Data containing LF (0x0A) which needs escaping
        data = bytes([0x00, 0x0A, 0x01])  # Second byte is LF

        adapter.write_binary("TEST ", data)

        # Get what was actually written
        written_data = self.mock_serial.write.call_args[0][0]

        # Should have: TEST #13<escaped_data>\r\n
        # Length header is #13 (3 bytes unescaped)
        # But actual data is escaped: 0x00, ESC, 0x0A, 0x01
        self.assertIn(b"#13", written_data)

        # Check that LF was escaped (ESC before LF)
        # Extract the data portion after header
        data_start = written_data.find(b"#13") + 3
        data_end = written_data.find(b"\r\n")
        actual_data = written_data[data_start:data_end]

        # Should be: 0x00, ESC, LF, 0x01
        self.assertEqual(actual_data, bytes([0x00, ESC, LF, 0x01]))

    def test_write_binary_all_special_chars(self):
        """Test escaping all special characters."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # Data with all special characters
        data = bytes([LF, CR, ESC, PLUS])

        adapter.write_binary("CMD ", data)

        written_data = self.mock_serial.write.call_args[0][0]

        # Extract escaped data
        data_start = written_data.find(b"#14") + 3
        data_end = written_data.find(b"\r\n")
        actual_data = written_data[data_start:data_end]

        # Each special char should be preceded by ESC
        expected = bytes([ESC, LF, ESC, CR, ESC, ESC, ESC, PLUS])
        self.assertEqual(actual_data, expected)

    def test_write_binary_large_data(self):
        """Test binary write with large data (multi-digit length)."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)
        self.mock_serial.write.reset_mock()

        # 1000 bytes of data
        data = bytes(range(256)) * 4  # 1024 bytes

        adapter.write_binary("DATA ", data)

        written_data = self.mock_serial.write.call_args[0][0]

        # Should have #4 prefix (4 digits in "1024")
        # But actual data will be longer due to escaping special chars
        self.assertIn(b"#41024", written_data)

    def test_read_binary(self):
        """Test reading binary data in IEEE 488.2 format."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        # Mock binary response: #14<4 bytes of data>
        header = b"#14"
        data = bytes([0x01, 0x02, 0x03, 0x04])
        self.mock_serial.read.side_effect = [
            header[:2],    # '#1'
            header[2:],    # '4'
            data           # actual data
        ]

        result = adapter.read_binary()

        self.assertEqual(result, data)

    def test_read_binary_chunked(self):
        """Test reading large binary data in chunks."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        # 8 bytes of data, read in 4-byte chunks
        data = bytes([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])

        self.mock_serial.read.side_effect = [
            b"#1",                        # Header start
            b"8",                          # Length digit
            data[:4],                      # First chunk
            data[4:]                       # Second chunk
        ]

        result = adapter.read_binary(chunk_size=4)

        self.assertEqual(result, data)

    def test_read_binary_invalid_header(self):
        """Test error handling for invalid binary header."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        self.mock_serial.read.return_value = b"XX"  # Invalid header

        with self.assertRaises(ValueError) as cm:
            adapter.read_binary()

        self.assertIn("Invalid binary block header", str(cm.exception))

    def test_close(self):
        """Test closing the adapter."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        adapter.close()

        self.mock_serial.close.assert_called_once()

    def test_context_manager(self):
        """Test using adapter as context manager."""
        with PrologixAdapter(port="COM4", gpib_address=4) as adapter:
            self.assertIsNotNone(adapter)

        # Verify close was called on exit
        self.mock_serial.close.assert_called()

    def test_repr(self):
        """Test string representation."""
        adapter = PrologixAdapter(port="COM4", gpib_address=4)

        repr_str = repr(adapter)

        self.assertIn("COM4", repr_str)
        self.assertIn("4", repr_str)


if __name__ == '__main__':
    unittest.main()
