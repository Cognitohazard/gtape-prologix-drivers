"""Unit tests for HP34401A multimeter driver."""

import pytest
from unittest.mock import patch


class TestHP34401A:
    """Test cases for HP34401A multimeter driver."""

    def test_initialization(self, dmm, mock_adapter):
        """Test DMM initialization."""
        assert dmm.adapter == mock_adapter

    def test_reset(self, dmm, mock_adapter):
        """Test DMM reset."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.reset()

        write_calls = [call[0][0] for call in mock_adapter.write.call_args_list]
        assert "*RST" in write_calls
        assert "*CLS" in write_calls
        mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_configure_dc_voltage_default(self, dmm, mock_adapter):
        """Test configuring DC voltage with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_voltage()

        mock_adapter.write.assert_called_with("CONF:VOLT:DC 10")

    def test_configure_dc_voltage_custom_range(self, dmm, mock_adapter):
        """Test configuring DC voltage with custom range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_voltage(range_volts=100)

        mock_adapter.write.assert_called_with("CONF:VOLT:DC 100")

    def test_configure_dc_voltage_with_resolution(self, dmm, mock_adapter):
        """Test configuring DC voltage with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)

        mock_adapter.write.assert_called_with("CONF:VOLT:DC 10,1e-05")

    def test_configure_ac_voltage_default(self, dmm, mock_adapter):
        """Test configuring AC voltage with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_ac_voltage()

        mock_adapter.write.assert_called_with("CONF:VOLT:AC 10")

    def test_configure_ac_voltage_with_resolution(self, dmm, mock_adapter):
        """Test configuring AC voltage with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_ac_voltage(range_volts=100, resolution=0.0001)

        mock_adapter.write.assert_called_with("CONF:VOLT:AC 100,0.0001")

    def test_configure_dc_current_default(self, dmm, mock_adapter):
        """Test configuring DC current with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_current()

        mock_adapter.write.assert_called_with("CONF:CURR:DC 1")

    def test_configure_dc_current_custom_range(self, dmm, mock_adapter):
        """Test configuring DC current with custom range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_current(range_amps=3)

        mock_adapter.write.assert_called_with("CONF:CURR:DC 3")

    def test_configure_dc_current_with_resolution(self, dmm, mock_adapter):
        """Test configuring DC current with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_dc_current(range_amps=1, resolution=0.000001)

        mock_adapter.write.assert_called_with("CONF:CURR:DC 1,1e-06")

    def test_configure_ac_current_default(self, dmm, mock_adapter):
        """Test configuring AC current with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_ac_current()

        mock_adapter.write.assert_called_with("CONF:CURR:AC 1")

    def test_configure_ac_current_with_resolution(self, dmm, mock_adapter):
        """Test configuring AC current with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_ac_current(range_amps=0.1, resolution=0.0000001)

        mock_adapter.write.assert_called_with("CONF:CURR:AC 0.1,1e-07")

    def test_configure_resistance_default(self, dmm, mock_adapter):
        """Test configuring resistance with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_resistance()

        mock_adapter.write.assert_called_with("CONF:RES 1000")

    def test_configure_resistance_custom_range(self, dmm, mock_adapter):
        """Test configuring resistance with custom range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_resistance(range_ohms=10000)

        mock_adapter.write.assert_called_with("CONF:RES 10000")

    def test_configure_resistance_with_resolution(self, dmm, mock_adapter):
        """Test configuring resistance with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_resistance(range_ohms=1000, resolution=0.001)

        mock_adapter.write.assert_called_with("CONF:RES 1000,0.001")

    def test_configure_resistance_4wire_default(self, dmm, mock_adapter):
        """Test configuring 4-wire resistance with default parameters."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_resistance_4wire()

        mock_adapter.write.assert_called_with("CONF:FRES 1000")

    def test_configure_resistance_4wire_with_resolution(self, dmm, mock_adapter):
        """Test configuring 4-wire resistance with resolution."""
        mock_adapter.ask.return_value = '+0,"No error"'

        dmm.configure_resistance_4wire(range_ohms=100, resolution=0.0001)

        mock_adapter.write.assert_called_with("CONF:FRES 100,0.0001")

    def test_read(self, dmm, mock_adapter):
        """Test taking a reading."""
        mock_adapter.ask.return_value = "+5.013456"

        reading = dmm.read()

        mock_adapter.ask.assert_called_with("READ?")
        assert reading == pytest.approx(5.013456)

    def test_read_negative(self, dmm, mock_adapter):
        """Test reading negative value."""
        mock_adapter.ask.return_value = "-2.345678"

        reading = dmm.read()

        assert reading == pytest.approx(-2.345678)

    def test_measure_voltage_dc(self, dmm, mock_adapter):
        """Test DC voltage measurement (autorange)."""
        mock_adapter.ask.return_value = "+12.345678"

        voltage = dmm.measure_voltage()

        mock_adapter.ask.assert_called_with("MEAS:VOLT:DC?")
        assert voltage == pytest.approx(12.345678)

    def test_measure_voltage_ac(self, dmm, mock_adapter):
        """Test AC voltage measurement (autorange)."""
        mock_adapter.ask.return_value = "+120.456789"

        voltage = dmm.measure_voltage(ac=True)

        mock_adapter.ask.assert_called_with("MEAS:VOLT:AC?")
        assert voltage == pytest.approx(120.456789)

    def test_measure_current_dc(self, dmm, mock_adapter):
        """Test DC current measurement (autorange)."""
        mock_adapter.ask.return_value = "+0.123456"

        current = dmm.measure_current()

        mock_adapter.ask.assert_called_with("MEAS:CURR:DC?")
        assert current == pytest.approx(0.123456)

    def test_measure_current_ac(self, dmm, mock_adapter):
        """Test AC current measurement (autorange)."""
        mock_adapter.ask.return_value = "+1.234567"

        current = dmm.measure_current(ac=True)

        mock_adapter.ask.assert_called_with("MEAS:CURR:AC?")
        assert current == pytest.approx(1.234567)

    def test_measure_resistance_2wire(self, dmm, mock_adapter):
        """Test 2-wire resistance measurement (autorange)."""
        mock_adapter.ask.return_value = "+1234.56789"

        resistance = dmm.measure_resistance()

        mock_adapter.ask.assert_called_with("MEAS:RES?")
        assert resistance == pytest.approx(1234.56789)

    def test_measure_resistance_4wire(self, dmm, mock_adapter):
        """Test 4-wire resistance measurement (autorange)."""
        mock_adapter.ask.return_value = "+999.888777"

        resistance = dmm.measure_resistance(four_wire=True)

        mock_adapter.ask.assert_called_with("MEAS:FRES?")
        assert resistance == pytest.approx(999.888777)

    def test_measure_frequency(self, dmm, mock_adapter):
        """Test frequency measurement (autorange)."""
        mock_adapter.ask.return_value = "+1000.123"

        frequency = dmm.measure_frequency()

        mock_adapter.ask.assert_called_with("MEAS:FREQ?")
        assert frequency == pytest.approx(1000.123)

    def test_check_errors_no_error(self, dmm, mock_adapter):
        """Test error checking with no errors."""
        mock_adapter.ask.return_value = '+0,"No error"'

        result = dmm.check_errors()

        assert result == '+0,"No error"'
        mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_check_errors_alternative_format(self, dmm, mock_adapter):
        """Test error checking with alternative no-error format."""
        mock_adapter.ask.return_value = '0,"No error"'

        result = dmm.check_errors()

        assert result == '0,"No error"'

    def test_check_errors_with_error(self, dmm, mock_adapter):
        """Test error checking when error exists."""
        mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = dmm.check_errors()

        assert result == '-113,"Undefined header"'
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error" in str(call) for call in print_calls)

    def test_scientific_notation(self, dmm, mock_adapter):
        """Test reading values in scientific notation."""
        mock_adapter.ask.return_value = "+1.23456E-03"

        reading = dmm.read()

        assert reading == pytest.approx(0.00123456)

    def test_configure_and_read_workflow(self, dmm, mock_adapter):
        """Test complete configure and read workflow."""
        mock_adapter.ask.side_effect = [
            '+0,"No error"',  # Error check after configure
            "+5.123456"       # Read value
        ]

        dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)
        reading = dmm.read()

        mock_adapter.write.assert_called_with("CONF:VOLT:DC 10,1e-05")
        assert reading == pytest.approx(5.123456)

    def test_measure_voltage_with_range(self, dmm, mock_adapter):
        """Test voltage measurement with fixed range."""
        mock_adapter.ask.return_value = "+4.567890"

        voltage = dmm.measure_voltage(range_volts=10)

        mock_adapter.ask.assert_called_with("MEAS:VOLT:DC? 10")
        assert voltage == pytest.approx(4.56789)

    def test_measure_current_with_range(self, dmm, mock_adapter):
        """Test current measurement with fixed range."""
        mock_adapter.ask.return_value = "+0.250000"

        current = dmm.measure_current(range_amps=1)

        mock_adapter.ask.assert_called_with("MEAS:CURR:DC? 1")
        assert current == pytest.approx(0.25)

    def test_measure_resistance_with_range(self, dmm, mock_adapter):
        """Test resistance measurement with fixed range."""
        mock_adapter.ask.return_value = "+470.123"

        resistance = dmm.measure_resistance(range_ohms=1000)

        mock_adapter.ask.assert_called_with("MEAS:RES? 1000")
        assert resistance == pytest.approx(470.123)
