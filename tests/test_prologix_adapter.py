"""Unit tests for PrologixAdapter class."""

import pytest
from unittest.mock import MagicMock, call, patch
from gtape_prologix_drivers.adapter import (
    PrologixAdapter, LF, CR, ESC, PLUS, SPECIAL_CHARS,
    PROLOGIX_BAUD_RATE, PROLOGIX_READ_TIMEOUT_MS
)


class TestPrologixAdapter:
    """Test cases for PrologixAdapter."""

    def test_initialization(self, adapter, mock_serial):
        """Test adapter initialization and Prologix configuration."""
        # Verify Prologix configuration commands were sent
        expected_calls = [
            call("++mode 1\r\n".encode()),
            call("++addr 10\r\n".encode()),
            call("++auto 0\r\n".encode()),
            call("++eos 3\r\n".encode()),
            call("++eoi 1\r\n".encode()),
            call(f"++read_tmo_ms {PROLOGIX_READ_TIMEOUT_MS}\r\n".encode()),
        ]

        actual_calls = mock_serial.write.call_args_list
        assert len(actual_calls) == 6
        for expected, actual in zip(expected_calls, actual_calls):
            assert expected == actual

    def test_initialization_custom_timeout(self, mock_serial):
        """Test adapter initialization with custom timeout."""
        with patch('gtape_prologix_drivers.adapter.serial.Serial', return_value=mock_serial):
            adapter = PrologixAdapter(port="COM3", gpib_address=22, timeout=5.0)
            assert adapter.port == "COM3"
            assert adapter.address == 22

    def test_switch_address(self, adapter, mock_serial):
        """Test switching GPIB address."""
        mock_serial.write.reset_mock()

        adapter.switch_address(5)

        assert adapter.address == 5
        mock_serial.write.assert_called_with("++addr 5\r\n".encode())

    def test_switch_address_same(self, adapter, mock_serial):
        """Test switching to same address (should be no-op)."""
        mock_serial.write.reset_mock()

        adapter.switch_address(10)  # Same as initial address

        mock_serial.write.assert_not_called()
        assert adapter.address == 10

    def test_write_command(self, adapter, mock_serial):
        """Test writing SCPI command."""
        mock_serial.write.reset_mock()

        adapter.write("*IDN?")

        mock_serial.write.assert_called_once_with("*IDN?\r\n".encode())

    def test_write_command_custom_delay(self, adapter, mock_serial):
        """Test write with custom delay."""
        mock_serial.write.reset_mock()

        with patch('gtape_prologix_drivers.adapter.time.sleep') as mock_sleep:
            adapter.write("VOLT 5.0", delay=0.5)
            mock_sleep.assert_called_with(0.5)

    def test_read(self, adapter, mock_serial):
        """Test reading response from instrument."""
        mock_serial.write.reset_mock()
        mock_serial.readline.return_value = b"HEWLETT-PACKARD,33120A,0,10.0\n\x00"

        response = adapter.read()

        mock_serial.write.assert_called_with("++read eoi\r\n".encode())
        assert response == "HEWLETT-PACKARD,33120A,0,10.0"

    def test_read_with_null_bytes(self, adapter, mock_serial):
        """Test reading response with null bytes."""
        mock_serial.readline.return_value = b"\x00+5.01378400E+00\n\x00"

        response = adapter.read()

        assert response == "+5.01378400E+00"

    def test_ask(self, adapter, mock_serial):
        """Test query (write + read)."""
        mock_serial.write.reset_mock()
        mock_serial.readline.return_value = b"+0,\"No error\"\n"

        response = adapter.ask("SYST:ERR?")

        calls = mock_serial.write.call_args_list
        assert calls[0] == call("SYST:ERR?\r\n".encode())
        assert calls[1] == call("++read eoi\r\n".encode())
        assert response == '+0,"No error"'

    def test_read_line(self, adapter, mock_serial):
        """Test read_line method."""
        mock_serial.write.reset_mock()
        mock_serial.readline.return_value = b"test response\n"

        response = adapter.read_line()

        mock_serial.write.assert_called_with("++read eoi\r\n".encode())
        assert response == "test response"

    def test_write_binary_simple(self, adapter, mock_serial):
        """Test writing binary data with IEEE 488.2 format."""
        mock_serial.write.reset_mock()

        data = bytes([0, 1, 2, 3])
        adapter.write_binary("DATA:DAC VOLATILE, ", data)

        expected = b"DATA:DAC VOLATILE, #14" + data + b"\r\n"
        mock_serial.write.assert_called_once_with(expected)

    def test_write_binary_with_escaping(self, adapter, mock_serial):
        """Test binary data with special characters requiring escaping."""
        mock_serial.write.reset_mock()

        data = bytes([0x00, 0x0A, 0x01])  # Second byte is LF
        adapter.write_binary("TEST ", data)

        written_data = mock_serial.write.call_args[0][0]
        assert b"#13" in written_data

        data_start = written_data.find(b"#13") + 3
        data_end = written_data.find(b"\r\n")
        actual_data = written_data[data_start:data_end]
        assert actual_data == bytes([0x00, ESC, LF, 0x01])

    def test_write_binary_all_special_chars(self, adapter, mock_serial):
        """Test escaping all special characters."""
        mock_serial.write.reset_mock()

        data = bytes([LF, CR, ESC, PLUS])
        adapter.write_binary("CMD ", data)

        written_data = mock_serial.write.call_args[0][0]
        data_start = written_data.find(b"#14") + 3
        data_end = written_data.find(b"\r\n")
        actual_data = written_data[data_start:data_end]

        expected = bytes([ESC, LF, ESC, CR, ESC, ESC, ESC, PLUS])
        assert actual_data == expected

    def test_write_binary_large_data(self, adapter, mock_serial):
        """Test binary write with large data (multi-digit length)."""
        mock_serial.write.reset_mock()

        data = bytes(range(256)) * 4  # 1024 bytes
        adapter.write_binary("DATA ", data)

        written_data = mock_serial.write.call_args[0][0]
        assert b"#41024" in written_data

    def test_read_binary(self, adapter, mock_serial):
        """Test reading binary data in IEEE 488.2 format."""
        header = b"#14"
        data = bytes([0x01, 0x02, 0x03, 0x04])
        mock_serial.read.side_effect = [
            header[:2],    # '#1'
            header[2:],    # '4'
            data           # actual data
        ]

        result = adapter.read_binary()

        assert result == data

    def test_read_binary_sends_read_command(self, adapter, mock_serial):
        """Test that read_binary sends ++read eoi command."""
        mock_serial.write.reset_mock()
        mock_serial.read.side_effect = [b"#1", b"4", bytes([0x01, 0x02, 0x03, 0x04])]

        adapter.read_binary()

        mock_serial.write.assert_called_with("++read eoi\r\n".encode())

    def test_read_binary_chunked(self, adapter, mock_serial):
        """Test reading large binary data in chunks."""
        data = bytes([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])

        mock_serial.read.side_effect = [
            b"#1",
            b"8",
            data[:4],
            data[4:]
        ]

        result = adapter.read_binary(chunk_size=4)

        assert result == data

    def test_read_binary_invalid_header(self, adapter, mock_serial):
        """Test error handling for invalid binary header."""
        mock_serial.read.return_value = b"XX"

        with pytest.raises(ValueError) as excinfo:
            adapter.read_binary()

        assert "Invalid binary block header" in str(excinfo.value)

    def test_read_binary_timeout_partial_header(self, adapter, mock_serial):
        """Test timeout error when header read is incomplete."""
        mock_serial.read.return_value = b"#"  # Only 1 byte instead of 2

        with pytest.raises(TimeoutError) as excinfo:
            adapter.read_binary()

        assert "Timeout reading binary header" in str(excinfo.value)

    def test_read_binary_timeout_partial_length(self, adapter, mock_serial):
        """Test timeout error when length field read is incomplete."""
        mock_serial.read.side_effect = [
            b"#3",     # Header says 3-digit length
            b"10",     # Only 2 bytes instead of 3
        ]

        with pytest.raises(TimeoutError) as excinfo:
            adapter.read_binary()

        assert "Timeout reading length field" in str(excinfo.value)

    def test_read_binary_timeout_partial_data(self, adapter, mock_serial):
        """Test timeout error when data read is incomplete."""
        mock_serial.read.side_effect = [
            b"#1",     # Header
            b"8",      # 8 bytes expected
            b"\x01\x02\x03\x04",  # First chunk (4 bytes)
            b"",       # Timeout - empty read
        ]

        with pytest.raises(TimeoutError) as excinfo:
            adapter.read_binary(chunk_size=4)

        assert "Timeout reading binary data" in str(excinfo.value)
        assert "4/8 bytes" in str(excinfo.value)

    def test_read_binary_leading_null_bytes(self, adapter, mock_serial):
        """Test that leading null bytes are skipped correctly."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        mock_serial.read.side_effect = [
            b"\x00#",  # Null byte followed by #
            b"1",      # Read next byte to get digit
            b"4",      # Length
            data,
        ]

        result = adapter.read_binary()

        assert result == data

    def test_read_binary_digit_count_zero_error(self, adapter, mock_serial):
        """Test error when IEEE header has 0 digit count."""
        mock_serial.read.side_effect = [b"#0"]

        with pytest.raises(ValueError) as excinfo:
            adapter.read_binary()

        assert "digit count is 0" in str(excinfo.value)

    def test_close(self, adapter, mock_serial):
        """Test closing the adapter."""
        adapter.close()

        mock_serial.close.assert_called_once()

    def test_close_idempotent(self, adapter, mock_serial):
        """Test closing the adapter multiple times is safe."""
        adapter.close()
        adapter.close()  # Should not raise

        # ser should be None after first close
        assert adapter.ser is None

    def test_context_manager(self, mock_serial):
        """Test using adapter as context manager."""
        with patch('gtape_prologix_drivers.adapter.serial.Serial', return_value=mock_serial):
            with PrologixAdapter(port="COM4", gpib_address=4) as adapter:
                assert adapter is not None

        mock_serial.close.assert_called()

    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)

        assert "COM1" in repr_str
        assert "10" in repr_str

    def test_verify_connection_success(self, adapter, mock_serial):
        """Test verify_connection when Prologix responds."""
        mock_serial.readline.return_value = b"Prologix GPIB-USB Controller version 6.0\n"

        result = adapter.verify_connection()

        mock_serial.write.assert_called_with("++ver\r\n".encode())
        assert result is True

    def test_verify_connection_failure(self, adapter, mock_serial):
        """Test verify_connection when no response."""
        mock_serial.readline.return_value = b""

        result = adapter.verify_connection()

        assert result is False

    def test_constants_defined(self):
        """Test that module constants are defined correctly."""
        assert LF == 0x0A
        assert CR == 0x0D
        assert ESC == 0x1B
        assert PLUS == 0x2B
        assert SPECIAL_CHARS == (LF, CR, ESC, PLUS)
        assert PROLOGIX_BAUD_RATE == 115200
        assert PROLOGIX_READ_TIMEOUT_MS == 4000
