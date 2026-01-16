"""Prologix GPIB-USB adapter for instrument control via pyserial.

Handles address switching, binary data transfer with IEEE 488.2 format,
and special character escaping (LF, CR, ESC, PLUS).

Valid Prologix ++ commands (firmware 6.x):
    ++addr 0-30         -- specify GPIB address
    ++addr              -- query GPIB address
    ++auto 0|1          -- enable (1) or disable (0) read-after-write
    ++auto              -- query read-after-write setting
    ++clr               -- issue device clear
    ++eoi 0|1           -- enable (1) or disable (0) EOI with last byte
    ++eoi               -- query eoi setting
    ++eos 0|1|2|3       -- EOS terminator - 0:CR+LF, 1:CR, 2:LF, 3:None
    ++eos               -- query eos setting
    ++eot_enable 0|1    -- enable (1) or disable (0) appending eot_char on EOI
    ++eot_enable        -- query eot_enable setting
    ++eot <char>        -- specify eot character in decimal
    ++eot_char          -- query eot_char character
    ++ifc               -- issue interface clear
    ++loc               -- set device to local
    ++mode 0|1          -- set mode - 0:DEVICE, 1:CONTROLLER
    ++mode              -- query current mode
    ++read [eoi|<char>] -- read until EOI, <char>, or timeout
    ++read_tmo_ms 500-4000 -- set read timeout in millisec
    ++read_tmo_ms       -- query timeout
    ++rst               -- reset controller
    ++spoll             -- serial poll currently addressed device
    ++spoll 0-30        -- serial poll device at specified address
    ++srq               -- query SRQ status
    ++trg               -- issue device trigger
    ++ver               -- query controller version
    ++help              -- display this help
"""

import serial
import time

# Prologix controller constants
PROLOGIX_BAUD_RATE = 115200
PROLOGIX_READ_TIMEOUT_MS = 4000
PROLOGIX_ADDRESS_SETTLE_MS = 20
PROLOGIX_CONFIG_DELAY = 0.1

# Special characters that must be escaped in binary data
LF = 0x0A    # Line Feed
CR = 0x0D    # Carriage Return
ESC = 0x1B   # Escape
PLUS = 0x2B  # Plus sign
SPECIAL_CHARS = (LF, CR, ESC, PLUS)


class PrologixAdapter:
    """Adapter for Prologix GPIB-USB controller using pyserial."""

    def __init__(self, port: str, gpib_address: int, timeout: float = 6.0, max_retries: int = 3):
        """Initialize Prologix adapter and configure controller.

        Args:
            port: Serial port (e.g., 'COM4')
            gpib_address: GPIB address of target instrument
            timeout: Serial timeout in seconds (default 6.0 for slow autorange measurements)
            max_retries: Number of reconnection attempts on serial errors (default 3)
        """
        self.port = port
        self.address = gpib_address
        self.timeout = timeout
        self.max_retries = max_retries
        self.ser: serial.Serial | None = serial.Serial(port, PROLOGIX_BAUD_RATE, timeout=timeout)
        self._configure_prologix()

    def _reconnect(self) -> None:
        """Attempt to reconnect to the serial port."""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        time.sleep(1.5)
        self.ser = serial.Serial(self.port, PROLOGIX_BAUD_RATE, timeout=self.timeout)
        self._configure_prologix()

    def _with_retry(self, operation, *args, **kwargs):
        """Execute operation with auto-reconnect on serial errors."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (serial.SerialException, OSError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"[GPIB] Serial error, reconnecting (attempt {attempt + 2}/{self.max_retries})...")
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        print(f"[GPIB] Reconnect failed: {reconnect_error}")
                        time.sleep(1.0)
        raise last_error

    def _configure_prologix(self) -> None:
        """Configure Prologix controller (mode, auto, eos, eoi settings)."""
        config_commands = [
            "++mode 1",                            # Controller mode
            f"++addr {self.address}",
            "++auto 0",                            # Manual read mode (prevents "Query Unterminated")
            "++eos 3",                             # No CR/LF appended
            "++eoi 1",                             # Assert EOI with last byte
            f"++read_tmo_ms {PROLOGIX_READ_TIMEOUT_MS}",  # 4s timeout (max) for slow measurements
        ]
        for cmd in config_commands:
            self.ser.write(f"{cmd}\r\n".encode())
            time.sleep(PROLOGIX_CONFIG_DELAY)
        # Flush any responses from config commands
        self.ser.reset_input_buffer()

    def verify_connection(self) -> bool:
        """Verify Prologix controller is responding.

        Returns True if controller responds correctly.
        """
        try:
            self.ser.write("++ver\r\n".encode())
            response = self.ser.readline().decode().strip()
            return "Prologix" in response
        except Exception:
            return False

    def switch_address(self, new_address: int) -> None:
        """Switch to a different GPIB address for multi-instrument control.

        Uses retry logic to handle transient serial errors. Only updates
        internal address state after successful write to prevent desync.
        """
        if new_address != self.address:
            self._with_retry(self._do_switch_address, new_address)
            self.address = new_address  # Only update after successful write

    def _do_switch_address(self, new_address: int) -> None:
        """Internal switch address implementation."""
        self.ser.write(f"++addr {new_address}\r\n".encode())
        time.sleep(PROLOGIX_ADDRESS_SETTLE_MS / 1000.0)  # 20ms settling time

    def _do_write(self, command: str, delay: float = 0) -> None:
        """Internal write implementation."""
        self.ser.write(f"{command}\r\n".encode())
        time.sleep(delay)

    def _do_read(self) -> str:
        """Internal read implementation."""
        self.ser.write("++read eoi\r\n".encode())
        return self.ser.readline().decode().strip(' \t\n\r\x00')

    def write(self, command: str, delay: float = 0) -> None:
        """Send SCPI command to instrument (with auto-reconnect)."""
        return self._with_retry(self._do_write, command, delay)

    def read(self) -> str:
        """Read response from instrument (with auto-reconnect).

        Blocks until instrument responds or serial timeout (default 6s).
        """
        return self._with_retry(self._do_read)

    def read_line(self) -> str:
        """Read a text line response from instrument (with auto-reconnect).

        Sends ++read eoi and reads one line. Use for text responses where
        you want retry protection (unlike direct serial access).
        """
        return self._with_retry(self._do_read)

    def ask(self, command: str) -> str:
        """Send query command and read response (with auto-reconnect)."""
        self.write(command)
        return self.read()

    def write_binary(self, command: str, data: bytes | list | tuple) -> None:
        """Send binary data with IEEE 488.2 block format (#<N><length><data>).

        Special chars (LF, CR, ESC, PLUS) are escaped automatically.
        Length field specifies unescaped byte count.
        """
        if isinstance(data, (list, tuple)):
            data = bytes(data)
        elif hasattr(data, 'tobytes'):
            data = data.tobytes()

        # Build IEEE 488.2 header
        data_length = len(data)
        length_str = str(data_length)
        header = f"#{len(length_str)}{length_str}"

        # Escape special characters
        escaped_data = bytearray()
        for byte in data:
            if byte in SPECIAL_CHARS:
                escaped_data.append(ESC)
            escaped_data.append(byte)

        full_command = command.encode() + header.encode() + bytes(escaped_data) + b"\r\n"
        self.ser.write(full_command)
        time.sleep(0.5)

    def read_binary(self, expected_bytes: int | None = None, chunk_size: int = 4096,
                    timeout_override: float | None = None) -> bytes:
        """Read binary data in IEEE 488.2 block format. Returns bytes.

        Sends ++read eoi to trigger the read, then parses IEEE 488.2 header.
        """
        old_timeout = self.ser.timeout

        if timeout_override:
            self.ser.timeout = timeout_override
        elif expected_bytes:
            # ~960 bytes/sec at 9600 baud, 2x safety margin, cap at 120s
            estimated_time = (expected_bytes / 960.0) * 2.0
            self.ser.timeout = min(estimated_time, 120.0)

        try:
            # Trigger the read from the instrument
            self.ser.write("++read eoi\r\n".encode())

            # Parse IEEE 488.2 header: #<N><length>
            # Some instruments send leading null bytes, so skip them
            header_start = self.ser.read(2)
            while header_start[0:1] == b'\x00':
                # Shift left and read one more byte
                header_start = header_start[1:] + self.ser.read(1)

            if header_start[0:1] != b'#':
                raise ValueError(f"Invalid binary block header: {header_start}")

            n = int(chr(header_start[1]))
            length_str = self.ser.read(n).decode()
            length = int(length_str)

            # Read data continuously (no delays - data is streaming)
            binary_data = bytearray()
            bytes_remaining = length
            while bytes_remaining > 0:
                chunk = self.ser.read(min(chunk_size, bytes_remaining))
                if len(chunk) == 0:
                    break
                binary_data.extend(chunk)
                bytes_remaining -= len(chunk)

            return bytes(binary_data)

        finally:
            self.ser.timeout = old_timeout

    def close(self) -> None:
        """Close serial port. Safe to call multiple times."""
        if self.ser is not None:
            try:
                if self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.ser = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return f"<PrologixAdapter(port='{self.port}', address={self.address})>"
