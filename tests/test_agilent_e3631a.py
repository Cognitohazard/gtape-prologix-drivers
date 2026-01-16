"""Unit tests for AgilentE3631A power supply driver."""

import unittest
from unittest.mock import Mock, MagicMock, call, patch
from gtape_prologix_drivers.instruments.agilent_e3631a import AgilentE3631A


class TestAgilentE3631A(unittest.TestCase):
    """Test cases for AgilentE3631A power supply driver."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.psu = AgilentE3631A(self.mock_adapter)

    def test_initialization(self):
        """Test PSU initialization."""
        self.assertEqual(self.psu.adapter, self.mock_adapter)
        self.assertIsNone(self.psu._current_channel)

    def test_channel_constants(self):
        """Test channel constant definitions."""
        self.assertEqual(AgilentE3631A.P6V, 1)
        self.assertEqual(AgilentE3631A.P25V, 2)
        self.assertEqual(AgilentE3631A.N25V, 3)

    def test_channel_specs(self):
        """Test channel specifications."""
        self.assertEqual(AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.P6V], (6.0, 5.0))
        self.assertEqual(AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.P25V], (25.0, 1.0))
        self.assertEqual(AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.N25V], (25.0, 1.0))

    def test_reset(self):
        """Test PSU reset."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.reset()

        # Verify *RST was sent
        self.mock_adapter.write.assert_called_with("*RST")

        # Verify error check was performed
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

        # Verify current channel was cleared
        self.assertIsNone(self.psu._current_channel)

    def test_select_channel_p6v(self):
        """Test selecting P6V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.select_channel(AgilentE3631A.P6V)

        self.mock_adapter.write.assert_called_with("INST:NSEL 1")
        self.assertEqual(self.psu._current_channel, AgilentE3631A.P6V)

    def test_select_channel_p25v(self):
        """Test selecting P25V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.select_channel(AgilentE3631A.P25V)

        self.mock_adapter.write.assert_called_with("INST:NSEL 2")
        self.assertEqual(self.psu._current_channel, AgilentE3631A.P25V)

    def test_select_channel_n25v(self):
        """Test selecting N25V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.select_channel(AgilentE3631A.N25V)

        self.mock_adapter.write.assert_called_with("INST:NSEL 3")
        self.assertEqual(self.psu._current_channel, AgilentE3631A.N25V)

    def test_select_channel_invalid(self):
        """Test error when selecting invalid channel."""
        with self.assertRaises(ValueError) as cm:
            self.psu.select_channel(5)

        self.assertIn("Invalid channel", str(cm.exception))

    def test_set_voltage_p6v_valid(self):
        """Test setting valid voltage on P6V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.psu._current_channel = AgilentE3631A.P6V

        self.psu.set_voltage(5.0)

        self.mock_adapter.write.assert_called_with("VOLT 5.0")

    def test_set_voltage_p25v_valid(self):
        """Test setting valid voltage on P25V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.psu._current_channel = AgilentE3631A.P25V

        self.psu.set_voltage(20.0)

        self.mock_adapter.write.assert_called_with("VOLT 20.0")

    def test_set_voltage_n25v_valid(self):
        """Test setting valid negative voltage on N25V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.psu._current_channel = AgilentE3631A.N25V

        self.psu.set_voltage(-15.0)

        self.mock_adapter.write.assert_called_with("VOLT -15.0")

    def test_set_voltage_no_channel_selected(self):
        """Test error when setting voltage without selecting channel."""
        with self.assertRaises(ValueError) as cm:
            self.psu.set_voltage(5.0)

        self.assertIn("No channel selected", str(cm.exception))

    def test_set_voltage_p6v_out_of_range_high(self):
        """Test error when P6V voltage too high."""
        self.psu._current_channel = AgilentE3631A.P6V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_voltage(7.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_voltage_p6v_out_of_range_negative(self):
        """Test error when P6V voltage negative."""
        self.psu._current_channel = AgilentE3631A.P6V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_voltage(-1.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_voltage_n25v_positive_error(self):
        """Test error when N25V channel given positive voltage."""
        self.psu._current_channel = AgilentE3631A.N25V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_voltage(10.0)

        self.assertIn("negative voltage", str(cm.exception))

    def test_set_voltage_n25v_out_of_range(self):
        """Test error when N25V voltage magnitude too large."""
        self.psu._current_channel = AgilentE3631A.N25V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_voltage(-30.0)

        self.assertIn("exceeds N25V range", str(cm.exception))

    def test_set_current_limit_p6v_valid(self):
        """Test setting valid current limit on P6V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.psu._current_channel = AgilentE3631A.P6V

        self.psu.set_current_limit(3.0)

        self.mock_adapter.write.assert_called_with("CURR 3.0")

    def test_set_current_limit_p25v_valid(self):
        """Test setting valid current limit on P25V channel."""
        self.mock_adapter.ask.return_value = '+0,"No error"'
        self.psu._current_channel = AgilentE3631A.P25V

        self.psu.set_current_limit(0.5)

        self.mock_adapter.write.assert_called_with("CURR 0.5")

    def test_set_current_limit_no_channel_selected(self):
        """Test error when setting current without selecting channel."""
        with self.assertRaises(ValueError) as cm:
            self.psu.set_current_limit(1.0)

        self.assertIn("No channel selected", str(cm.exception))

    def test_set_current_limit_p6v_out_of_range(self):
        """Test error when P6V current limit too high."""
        self.psu._current_channel = AgilentE3631A.P6V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_current_limit(6.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_current_limit_p25v_out_of_range(self):
        """Test error when P25V current limit too high."""
        self.psu._current_channel = AgilentE3631A.P25V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_current_limit(2.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_current_limit_negative(self):
        """Test error when current limit negative."""
        self.psu._current_channel = AgilentE3631A.P6V

        with self.assertRaises(ValueError) as cm:
            self.psu.set_current_limit(-1.0)

        self.assertIn("out of range", str(cm.exception))

    def test_enable_output_true(self):
        """Test enabling output."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.enable_output(True)

        self.mock_adapter.write.assert_called_with("OUTP 1")

    def test_enable_output_false(self):
        """Test disabling output."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.enable_output(False)

        self.mock_adapter.write.assert_called_with("OUTP 0")

    def test_enable_output_default(self):
        """Test enable_output default value (True)."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.enable_output()

        self.mock_adapter.write.assert_called_with("OUTP 1")

    def test_measure_voltage(self):
        """Test measuring voltage."""
        self.mock_adapter.ask.return_value = "+5.013"

        voltage = self.psu.measure_voltage()

        self.mock_adapter.ask.assert_called_with("MEAS:VOLT?")
        self.assertAlmostEqual(voltage, 5.013)

    def test_measure_current(self):
        """Test measuring current."""
        self.mock_adapter.ask.return_value = "+0.1234"

        current = self.psu.measure_current()

        self.mock_adapter.ask.assert_called_with("MEAS:CURR?")
        self.assertAlmostEqual(current, 0.1234)

    def test_check_errors_no_error(self):
        """Test error checking with no errors."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        result = self.psu.check_errors()

        self.assertEqual(result, '+0,"No error"')
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_check_errors_with_error(self):
        """Test error checking when error exists."""
        self.mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = self.psu.check_errors()

        self.assertEqual(result, '-113,"Undefined header"')

        # Should have printed the error
        print_calls = [str(call) for call in mock_print.call_args_list]
        error_found = any("Error" in str(call) for call in print_calls)
        self.assertTrue(error_found)

    def test_configure_output_complete(self):
        """Test complete configuration workflow."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.psu.configure_output(AgilentE3631A.P25V, 12.0, 0.5)

        # Verify all steps were performed
        # 1. Select channel
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("INST:NSEL 2" in str(call) for call in write_calls))

        # 2. Set voltage
        self.assertTrue(any("VOLT 12.0" in str(call) for call in write_calls))

        # 3. Set current
        self.assertTrue(any("CURR 0.5" in str(call) for call in write_calls))

        # Verify channel was set
        self.assertEqual(self.psu._current_channel, AgilentE3631A.P25V)


if __name__ == '__main__':
    unittest.main()
