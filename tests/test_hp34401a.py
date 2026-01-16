"""Unit tests for HP34401A multimeter driver."""

import unittest
from unittest.mock import Mock, MagicMock, call, patch
from gtape_prologix_drivers.instruments.hp34401a import HP34401A


class TestHP34401A(unittest.TestCase):
    """Test cases for HP34401A multimeter driver."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.dmm = HP34401A(self.mock_adapter)

    def test_initialization(self):
        """Test DMM initialization."""
        self.assertEqual(self.dmm.adapter, self.mock_adapter)

    def test_reset(self):
        """Test DMM reset."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.mock_adapter.ser = Mock()  # Mock serial port for buffer clear

        self.dmm.reset()

        # Verify *RST and *CLS were sent
        write_calls = [call[0][0] for call in self.mock_adapter.write.call_args_list]
        self.assertIn("*RST", write_calls)
        self.assertIn("*CLS", write_calls)

        # Verify error check was performed
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_configure_dc_voltage_default(self):
        """Test configuring DC voltage with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_voltage()

        # Should use default range (10V)
        self.mock_adapter.write.assert_called_with("CONF:VOLT:DC 10")

    def test_configure_dc_voltage_custom_range(self):
        """Test configuring DC voltage with custom range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_voltage(range_volts=100)

        self.mock_adapter.write.assert_called_with("CONF:VOLT:DC 100")

    def test_configure_dc_voltage_with_resolution(self):
        """Test configuring DC voltage with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)

        self.mock_adapter.write.assert_called_with("CONF:VOLT:DC 10,1e-05")

    def test_configure_ac_voltage_default(self):
        """Test configuring AC voltage with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_ac_voltage()

        self.mock_adapter.write.assert_called_with("CONF:VOLT:AC 10")

    def test_configure_ac_voltage_with_resolution(self):
        """Test configuring AC voltage with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_ac_voltage(range_volts=100, resolution=0.0001)

        self.mock_adapter.write.assert_called_with("CONF:VOLT:AC 100,0.0001")

    def test_configure_dc_current_default(self):
        """Test configuring DC current with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_current()

        self.mock_adapter.write.assert_called_with("CONF:CURR:DC 1")

    def test_configure_dc_current_custom_range(self):
        """Test configuring DC current with custom range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_current(range_amps=3)

        self.mock_adapter.write.assert_called_with("CONF:CURR:DC 3")

    def test_configure_dc_current_with_resolution(self):
        """Test configuring DC current with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_dc_current(range_amps=1, resolution=0.000001)

        self.mock_adapter.write.assert_called_with("CONF:CURR:DC 1,1e-06")

    def test_configure_ac_current_default(self):
        """Test configuring AC current with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_ac_current()

        self.mock_adapter.write.assert_called_with("CONF:CURR:AC 1")

    def test_configure_ac_current_with_resolution(self):
        """Test configuring AC current with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_ac_current(range_amps=0.1, resolution=0.0000001)

        self.mock_adapter.write.assert_called_with("CONF:CURR:AC 0.1,1e-07")

    def test_configure_resistance_default(self):
        """Test configuring resistance with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_resistance()

        self.mock_adapter.write.assert_called_with("CONF:RES 1000")

    def test_configure_resistance_custom_range(self):
        """Test configuring resistance with custom range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_resistance(range_ohms=10000)

        self.mock_adapter.write.assert_called_with("CONF:RES 10000")

    def test_configure_resistance_with_resolution(self):
        """Test configuring resistance with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_resistance(range_ohms=1000, resolution=0.001)

        self.mock_adapter.write.assert_called_with("CONF:RES 1000,0.001")

    def test_configure_resistance_4wire_default(self):
        """Test configuring 4-wire resistance with default parameters."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_resistance_4wire()

        self.mock_adapter.write.assert_called_with("CONF:FRES 1000")

    def test_configure_resistance_4wire_with_resolution(self):
        """Test configuring 4-wire resistance with resolution."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.dmm.configure_resistance_4wire(range_ohms=100, resolution=0.0001)

        self.mock_adapter.write.assert_called_with("CONF:FRES 100,0.0001")

    def test_read(self):
        """Test taking a reading (after configure, no delay needed)."""
        self.mock_adapter.ask.return_value = "+5.013456"

        reading = self.dmm.read()

        self.mock_adapter.ask.assert_called_with("READ?")
        self.assertAlmostEqual(reading, 5.013456)

    def test_read_negative(self):
        """Test reading negative value."""
        self.mock_adapter.ask.return_value = "-2.345678"

        reading = self.dmm.read()

        self.assertAlmostEqual(reading, -2.345678)

    def test_measure_voltage_dc(self):
        """Test DC voltage measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+12.345678"

        voltage = self.dmm.measure_voltage()

        self.mock_adapter.ask.assert_called_with("MEAS:VOLT:DC?")
        self.assertAlmostEqual(voltage, 12.345678)

    def test_measure_voltage_ac(self):
        """Test AC voltage measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+120.456789"

        voltage = self.dmm.measure_voltage(ac=True)

        self.mock_adapter.ask.assert_called_with("MEAS:VOLT:AC?")
        self.assertAlmostEqual(voltage, 120.456789)

    def test_measure_current_dc(self):
        """Test DC current measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+0.123456"

        current = self.dmm.measure_current()

        self.mock_adapter.ask.assert_called_with("MEAS:CURR:DC?")
        self.assertAlmostEqual(current, 0.123456)

    def test_measure_current_ac(self):
        """Test AC current measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+1.234567"

        current = self.dmm.measure_current(ac=True)

        self.mock_adapter.ask.assert_called_with("MEAS:CURR:AC?")
        self.assertAlmostEqual(current, 1.234567)

    def test_measure_resistance_2wire(self):
        """Test 2-wire resistance measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+1234.56789"

        resistance = self.dmm.measure_resistance()

        self.mock_adapter.ask.assert_called_with("MEAS:RES?")
        self.assertAlmostEqual(resistance, 1234.56789)

    def test_measure_resistance_4wire(self):
        """Test 4-wire resistance measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+999.888777"

        resistance = self.dmm.measure_resistance(four_wire=True)

        self.mock_adapter.ask.assert_called_with("MEAS:FRES?")
        self.assertAlmostEqual(resistance, 999.888777)

    def test_measure_frequency(self):
        """Test frequency measurement (autorange)."""
        self.mock_adapter.ask.return_value = "+1000.123"

        frequency = self.dmm.measure_frequency()

        self.mock_adapter.ask.assert_called_with("MEAS:FREQ?")
        self.assertAlmostEqual(frequency, 1000.123)

    def test_check_errors_no_error(self):
        """Test error checking with no errors."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        result = self.dmm.check_errors()

        self.assertEqual(result, '+0,"No error"')
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_check_errors_with_error(self):
        """Test error checking when error exists."""
        self.mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = self.dmm.check_errors()

        self.assertEqual(result, '-113,"Undefined header"')

        # Should have printed the error
        print_calls = [str(call) for call in mock_print.call_args_list]
        error_found = any("Error" in str(call) for call in print_calls)
        self.assertTrue(error_found)

    def test_scientific_notation(self):
        """Test reading values in scientific notation."""
        self.mock_adapter.ask.return_value = "+1.23456E-03"

        reading = self.dmm.read()

        self.assertAlmostEqual(reading, 0.00123456)

    def test_configure_and_read_workflow(self):
        """Test complete configure and read workflow."""
        self.mock_adapter.ask.side_effect = [
            '+0,"No error"',  # Error check after configure
            "+5.123456"       # Read value
        ]

        self.dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)
        reading = self.dmm.read()

        # Verify configuration
        self.mock_adapter.write.assert_called_with("CONF:VOLT:DC 10,1e-05")

        # Verify reading (no delay - range already configured)
        self.mock_adapter.ask.assert_called_with("READ?")
        self.assertAlmostEqual(reading, 5.123456)

    def test_measure_voltage_with_range(self):
        """Test voltage measurement with fixed range (fast, no delay)."""
        self.mock_adapter.ask.return_value = "+4.567890"

        voltage = self.dmm.measure_voltage(range_volts=10)

        self.mock_adapter.ask.assert_called_with("MEAS:VOLT:DC? 10")
        self.assertAlmostEqual(voltage, 4.56789)

    def test_measure_current_with_range(self):
        """Test current measurement with fixed range (fast, no delay)."""
        self.mock_adapter.ask.return_value = "+0.250000"

        current = self.dmm.measure_current(range_amps=1)

        self.mock_adapter.ask.assert_called_with("MEAS:CURR:DC? 1")
        self.assertAlmostEqual(current, 0.25)

    def test_measure_resistance_with_range(self):
        """Test resistance measurement with fixed range (fast, no delay)."""
        self.mock_adapter.ask.return_value = "+470.123"

        resistance = self.dmm.measure_resistance(range_ohms=1000)

        self.mock_adapter.ask.assert_called_with("MEAS:RES? 1000")
        self.assertAlmostEqual(resistance, 470.123)


if __name__ == '__main__':
    unittest.main()
