"""Unit tests for PLZ164W electronic load driver."""

import unittest
from unittest.mock import Mock, patch
from gtape_prologix_drivers.instruments.plz164w import PLZ164W


class TestPLZ164W(unittest.TestCase):
    """Test cases for PLZ164W electronic load driver."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.load = PLZ164W(self.mock_adapter)

    def test_initialization(self):
        """Test load initialization."""
        self.assertEqual(self.load.adapter, self.mock_adapter)
        self.assertIsNone(self.load._current_mode)

    def test_mode_constants(self):
        """Test mode constant definitions."""
        self.assertEqual(PLZ164W.MODE_CC, "CURR")
        self.assertEqual(PLZ164W.MODE_CV, "VOLT")
        self.assertEqual(PLZ164W.MODE_CR, "RES")
        self.assertEqual(PLZ164W.MODE_CP, "POW")
        self.assertEqual(PLZ164W.MODE_CCCV, "CCCV")
        self.assertEqual(PLZ164W.MODE_CRCV, "CRCV")

    def test_range_constants(self):
        """Test range constant definitions."""
        self.assertEqual(PLZ164W.CURR_RANGE_LOW, "LOW")
        self.assertEqual(PLZ164W.CURR_RANGE_MED, "MEDIUM")
        self.assertEqual(PLZ164W.CURR_RANGE_HIGH, "HIGH")
        self.assertEqual(PLZ164W.VOLT_RANGE_LOW, "LOW")
        self.assertEqual(PLZ164W.VOLT_RANGE_HIGH, "HIGH")

    def test_specification_constants(self):
        """Test specification constants."""
        self.assertEqual(PLZ164W.VOLTAGE_MIN, 1.5)
        self.assertEqual(PLZ164W.VOLTAGE_MAX, 150.0)
        self.assertEqual(PLZ164W.CURRENT_MAX, 33.0)
        self.assertEqual(PLZ164W.POWER_MAX, 165.0)
        self.assertEqual(PLZ164W.RESISTANCE_MIN, 0.5)
        self.assertEqual(PLZ164W.RESISTANCE_MAX, 6000.0)

    def test_reset(self):
        """Test load reset."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.reset()

        # Verify *RST was sent
        self.mock_adapter.write.assert_called()
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("*RST" in str(call) for call in write_calls))

        # Verify error check was performed
        self.mock_adapter.ask.assert_called_with("SYSTem:ERRor?")

        # Verify current mode was cleared
        self.assertIsNone(self.load._current_mode)

    def test_get_identification(self):
        """Test querying instrument identification."""
        self.mock_adapter.ask.return_value = "KIKUSUI,PLZ164W,12345,1.00"

        idn = self.load.get_identification()

        self.mock_adapter.ask.assert_called_with("*IDN?")
        self.assertEqual(idn, "KIKUSUI,PLZ164W,12345,1.00")

    def test_set_mode_cc(self):
        """Test setting CC mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_mode(PLZ164W.MODE_CC)

        self.mock_adapter.write.assert_called_with("SOURce:FUNCtion CURR")
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CC)

    def test_set_mode_cv(self):
        """Test setting CV mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_mode(PLZ164W.MODE_CV)

        self.mock_adapter.write.assert_called_with("SOURce:FUNCtion VOLT")
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CV)

    def test_set_mode_cr(self):
        """Test setting CR mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_mode(PLZ164W.MODE_CR)

        self.mock_adapter.write.assert_called_with("SOURce:FUNCtion RES")
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CR)

    def test_set_mode_cp(self):
        """Test setting CP mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_mode(PLZ164W.MODE_CP)

        self.mock_adapter.write.assert_called_with("SOURce:FUNCtion POW")
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CP)

    def test_set_mode_invalid(self):
        """Test error when setting invalid mode."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_mode("INVALID")

        self.assertIn("Invalid mode", str(cm.exception))

    def test_get_mode(self):
        """Test querying current mode."""
        self.mock_adapter.ask.return_value = "CURR"

        mode = self.load.get_mode()

        self.mock_adapter.ask.assert_called_with("SOURce:FUNCtion?")
        self.assertEqual(mode, "CURR")

    def test_set_current_valid(self):
        """Test setting valid current."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_current(5.0)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent 5.0")

    def test_set_current_max(self):
        """Test setting maximum current."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_current(33.0)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent 33.0")

    def test_set_current_out_of_range_high(self):
        """Test error when current too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_current(40.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_current_negative(self):
        """Test error when current negative."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_current(-1.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_current(self):
        """Test querying current setting."""
        self.mock_adapter.ask.return_value = "2.500"

        current = self.load.get_current()

        self.mock_adapter.ask.assert_called_with("SOURce:CURRent?")
        self.assertAlmostEqual(current, 2.5)

    def test_set_voltage_valid(self):
        """Test setting valid voltage."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_voltage(12.0)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage 12.0")

    def test_set_voltage_max(self):
        """Test setting maximum voltage."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_voltage(150.0)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage 150.0")

    def test_set_voltage_out_of_range_high(self):
        """Test error when voltage too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_voltage(200.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_voltage_out_of_range_low(self):
        """Test error when voltage too low."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_voltage(1.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_voltage(self):
        """Test querying voltage setting."""
        self.mock_adapter.ask.return_value = "12.000"

        voltage = self.load.get_voltage()

        self.mock_adapter.ask.assert_called_with("SOURce:VOLTage?")
        self.assertAlmostEqual(voltage, 12.0)

    def test_set_power_valid(self):
        """Test setting valid power."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_power(50.0)

        self.mock_adapter.write.assert_called_with("SOURce:POWer 50.0")

    def test_set_power_max(self):
        """Test setting maximum power."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_power(165.0)

        self.mock_adapter.write.assert_called_with("SOURce:POWer 165.0")

    def test_set_power_out_of_range_high(self):
        """Test error when power too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_power(200.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_power_negative(self):
        """Test error when power negative."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_power(-10.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_power(self):
        """Test querying power setting."""
        self.mock_adapter.ask.return_value = "50.000"

        power = self.load.get_power()

        self.mock_adapter.ask.assert_called_with("SOURce:POWer?")
        self.assertAlmostEqual(power, 50.0)

    def test_set_resistance_valid(self):
        """Test setting valid resistance."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_resistance(100.0)

        # Should convert to conductance (1/100 = 0.01)
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("SOURce:CONDuctance 0.01" in str(call) for call in write_calls))

    def test_set_resistance_min(self):
        """Test setting minimum resistance."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_resistance(0.5)

        # Should convert to conductance (1/0.5 = 2.0)
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("SOURce:CONDuctance 2.0" in str(call) for call in write_calls))

    def test_set_resistance_out_of_range_high(self):
        """Test error when resistance too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_resistance(7000.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_resistance_out_of_range_low(self):
        """Test error when resistance too low."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_resistance(0.1)

        self.assertIn("out of range", str(cm.exception))

    def test_get_resistance(self):
        """Test querying resistance setting."""
        # Return conductance of 0.01 (should convert to 100Ω)
        self.mock_adapter.ask.return_value = "0.01"

        resistance = self.load.get_resistance()

        self.mock_adapter.ask.assert_called_with("SOURce:CONDuctance?")
        self.assertAlmostEqual(resistance, 100.0)

    def test_get_resistance_zero_conductance(self):
        """Test querying resistance when conductance is zero."""
        self.mock_adapter.ask.return_value = "0.0"

        resistance = self.load.get_resistance()

        self.assertEqual(resistance, float('inf'))

    def test_set_current_range_low(self):
        """Test setting low current range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_current_range(PLZ164W.CURR_RANGE_LOW)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:RANGe LOW")

    def test_set_current_range_high(self):
        """Test setting high current range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_current_range(PLZ164W.CURR_RANGE_HIGH)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:RANGe HIGH")

    def test_set_current_range_invalid(self):
        """Test error when setting invalid current range."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_current_range("INVALID")

        self.assertIn("Invalid current range", str(cm.exception))

    def test_set_voltage_range_low(self):
        """Test setting low voltage range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_voltage_range(PLZ164W.VOLT_RANGE_LOW)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage:RANGe LOW")

    def test_set_voltage_range_high(self):
        """Test setting high voltage range."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_voltage_range(PLZ164W.VOLT_RANGE_HIGH)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage:RANGe HIGH")

    def test_set_voltage_range_invalid(self):
        """Test error when setting invalid voltage range."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_voltage_range("INVALID")

        self.assertIn("Invalid voltage range", str(cm.exception))

    def test_enable_input_true(self):
        """Test enabling input."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.enable_input(True)

        self.mock_adapter.write.assert_called_with("INPut 1")

    def test_enable_input_false(self):
        """Test disabling input."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.enable_input(False)

        self.mock_adapter.write.assert_called_with("INPut 0")

    def test_enable_input_default(self):
        """Test enable_input default value (True)."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.enable_input()

        self.mock_adapter.write.assert_called_with("INPut 1")

    def test_get_input_state_enabled(self):
        """Test querying input state when enabled."""
        self.mock_adapter.ask.return_value = "1"

        state = self.load.get_input_state()

        self.mock_adapter.ask.assert_called_with("INPut?")
        self.assertTrue(state)

    def test_get_input_state_disabled(self):
        """Test querying input state when disabled."""
        self.mock_adapter.ask.return_value = "0"

        state = self.load.get_input_state()

        self.mock_adapter.ask.assert_called_with("INPut?")
        self.assertFalse(state)

    def test_set_short_mode_enable(self):
        """Test enabling short mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_short_mode(True)

        self.mock_adapter.write.assert_called_with("INPut:SHORt 1")

    def test_set_short_mode_disable(self):
        """Test disabling short mode."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_short_mode(False)

        self.mock_adapter.write.assert_called_with("INPut:SHORt 0")

    def test_set_short_mode_default(self):
        """Test set_short_mode default value (False)."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_short_mode()

        self.mock_adapter.write.assert_called_with("INPut:SHORt 0")

    def test_measure_voltage(self):
        """Test measuring voltage."""
        self.mock_adapter.ask.return_value = "12.345"

        voltage = self.load.measure_voltage()

        self.mock_adapter.ask.assert_called_with("MEASure:VOLTage?")
        self.assertAlmostEqual(voltage, 12.345)

    def test_measure_current(self):
        """Test measuring current."""
        self.mock_adapter.ask.return_value = "2.567"

        current = self.load.measure_current()

        self.mock_adapter.ask.assert_called_with("MEASure:CURRent?")
        self.assertAlmostEqual(current, 2.567)

    def test_measure_power(self):
        """Test measuring power."""
        self.mock_adapter.ask.return_value = "31.695"

        power = self.load.measure_power()

        self.mock_adapter.ask.assert_called_with("MEASure:POWer?")
        self.assertAlmostEqual(power, 31.695)

    def test_check_errors_no_error(self):
        """Test error checking with no errors."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        result = self.load.check_errors()

        self.assertEqual(result, '+0,"No error"')
        self.mock_adapter.ask.assert_called_with("SYSTem:ERRor?")

    def test_check_errors_alternative_format(self):
        """Test error checking with alternative no-error format."""
        self.mock_adapter.ask.return_value = '0,"No error"'

        result = self.load.check_errors()

        self.assertEqual(result, '0,"No error"')

    def test_check_errors_with_error(self):
        """Test error checking when error exists."""
        self.mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = self.load.check_errors()

        self.assertEqual(result, '-113,"Undefined header"')

        # Should have printed the error
        print_calls = [str(call) for call in mock_print.call_args_list]
        error_found = any("Error" in str(call) for call in print_calls)
        self.assertTrue(error_found)

    def test_configure_cc_mode(self):
        """Test complete CC mode configuration."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.configure_cc_mode(5.0, PLZ164W.CURR_RANGE_HIGH)

        # Verify all steps were performed
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]

        # 1. Set mode to CC
        self.assertTrue(any("SOURce:FUNCtion CURR" in str(call) for call in write_calls))

        # 2. Set current range
        self.assertTrue(any("SOURce:CURRent:RANGe HIGH" in str(call) for call in write_calls))

        # 3. Set current
        self.assertTrue(any("SOURce:CURRent 5.0" in str(call) for call in write_calls))

        # Verify mode was set
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CC)

    def test_configure_cc_mode_no_range(self):
        """Test CC mode configuration without range setting."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.configure_cc_mode(5.0)

        # Verify current was set but range was not
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]
        self.assertTrue(any("SOURce:CURRent 5.0" in str(call) for call in write_calls))
        self.assertFalse(any("RANG" in str(call) for call in write_calls))

    def test_configure_cv_mode(self):
        """Test complete CV mode configuration."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.configure_cv_mode(12.0, PLZ164W.VOLT_RANGE_HIGH)

        # Verify all steps were performed
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]

        # 1. Set mode to CV
        self.assertTrue(any("SOURce:FUNCtion VOLT" in str(call) for call in write_calls))

        # 2. Set voltage range
        self.assertTrue(any("SOURce:VOLTage:RANGe HIGH" in str(call) for call in write_calls))

        # 3. Set voltage
        self.assertTrue(any("SOURce:VOLTage 12.0" in str(call) for call in write_calls))

        # Verify mode was set
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CV)

    def test_configure_cr_mode(self):
        """Test complete CR mode configuration."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.configure_cr_mode(100.0)

        # Verify all steps were performed
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]

        # 1. Set mode to CR
        self.assertTrue(any("SOURce:FUNCtion RES" in str(call) for call in write_calls))

        # 2. Set resistance (as conductance)
        self.assertTrue(any("SOURce:CONDuctance" in str(call) for call in write_calls))

        # Verify mode was set
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CR)

    def test_configure_cp_mode(self):
        """Test complete CP mode configuration."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.configure_cp_mode(50.0)

        # Verify all steps were performed
        write_calls = [str(call) for call in self.mock_adapter.write.call_args_list]

        # 1. Set mode to CP
        self.assertTrue(any("SOURce:FUNCtion POW" in str(call) for call in write_calls))

        # 2. Set power
        self.assertTrue(any("SOURce:POWer 50.0" in str(call) for call in write_calls))

        # Verify mode was set
        self.assertEqual(self.load._current_mode, PLZ164W.MODE_CP)

    # --- Overpower Protection (OPP) Tests ---

    def test_opp_action_constants(self):
        """Test OPP action constant definitions.

        Note: Only LIMIT is supported via SCPI. LOAD OFF is front-panel only.
        """
        self.assertEqual(PLZ164W.OPP_ACTION_LIMIT, "LIM")

    def test_opp_max_constant(self):
        """Test OPP maximum constant."""
        self.assertEqual(PLZ164W.OPP_MAX, 181.5)

    def test_set_overpower_protection_valid(self):
        """Test setting valid OPP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overpower_protection(100.0, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 100.0")

    def test_set_overpower_protection_max(self):
        """Test setting maximum OPP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overpower_protection(181.5, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 181.5")

    def test_set_overpower_protection_zero(self):
        """Test setting OPP threshold to zero."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overpower_protection(0.0, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 0.0")

    def test_set_overpower_protection_out_of_range_high(self):
        """Test error when OPP threshold too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_overpower_protection(200.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_overpower_protection_negative(self):
        """Test error when OPP threshold negative."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_overpower_protection(-10.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_overpower_protection(self):
        """Test querying OPP threshold."""
        self.mock_adapter.ask.return_value = "150.000"

        opp = self.load.get_overpower_protection()

        self.mock_adapter.ask.assert_called_with("SOURce:POWer:PROTection?")
        self.assertAlmostEqual(opp, 150.0)

    def test_set_overpower_protection_action_limit(self):
        """Test setting OPP action to LIMIT."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overpower_protection_action(PLZ164W.OPP_ACTION_LIMIT)

        self.mock_adapter.write.assert_called_with("SOURce:POWer:PROTection:ACTion LIM")

    def test_set_overpower_protection_action_invalid(self):
        """Test error when setting non-LIM OPP action.

        Note: Only LIM is supported via SCPI. LOAD OFF requires front panel.
        """
        with self.assertRaises(ValueError) as cm:
            self.load.set_overpower_protection_action("OFF")

        self.assertIn("Only 'LIM' supported via SCPI", str(cm.exception))

    def test_get_overpower_protection_action(self):
        """Test querying OPP action setting."""
        self.mock_adapter.ask.return_value = "LIM"

        action = self.load.get_overpower_protection_action()

        self.mock_adapter.ask.assert_called_with("SOURce:POWer:PROTection:ACTion?")
        self.assertEqual(action, "LIM")

    def test_get_overpower_protection_action_off(self):
        """Test querying OPP action when set to OFF."""
        self.mock_adapter.ask.return_value = "OFF"

        action = self.load.get_overpower_protection_action()

        self.assertEqual(action, "OFF")

    # --- Overcurrent Protection (OCP) Tests ---

    def test_protection_action_constants(self):
        """Test protection action constant definitions.

        Note: Only LIMIT is supported via SCPI. LOAD OFF is front-panel only.
        """
        self.assertEqual(PLZ164W.PROT_ACTION_LIMIT, "LIM")

    def test_ocp_max_constant(self):
        """Test OCP maximum constant."""
        self.assertEqual(PLZ164W.OCP_MAX, 36.29)

    def test_set_overcurrent_protection_valid(self):
        """Test setting valid OCP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overcurrent_protection(10.0)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 10.0")

    def test_set_overcurrent_protection_max(self):
        """Test setting maximum OCP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overcurrent_protection(36.29)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 36.29")

    def test_set_overcurrent_protection_zero(self):
        """Test setting OCP threshold to zero."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overcurrent_protection(0.0)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 0.0")

    def test_set_overcurrent_protection_out_of_range_high(self):
        """Test error when OCP threshold too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_overcurrent_protection(40.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_overcurrent_protection_negative(self):
        """Test error when OCP threshold negative."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_overcurrent_protection(-5.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_overcurrent_protection(self):
        """Test querying OCP threshold."""
        self.mock_adapter.ask.return_value = "10.000"

        ocp = self.load.get_overcurrent_protection()

        self.mock_adapter.ask.assert_called_with("SOURce:CURRent:PROTection?")
        self.assertAlmostEqual(ocp, 10.0)

    def test_set_overcurrent_protection_action_limit(self):
        """Test setting OCP action to LIMIT."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_overcurrent_protection_action(PLZ164W.PROT_ACTION_LIMIT)

        self.mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection:ACTion LIM")

    def test_set_overcurrent_protection_action_invalid(self):
        """Test error when setting non-LIM OCP action.

        Note: Only LIM is supported via SCPI. LOAD OFF requires front panel.
        """
        with self.assertRaises(ValueError) as cm:
            self.load.set_overcurrent_protection_action("OFF")

        self.assertIn("Only 'LIM' supported via SCPI", str(cm.exception))

    def test_get_overcurrent_protection_action(self):
        """Test querying OCP action setting."""
        self.mock_adapter.ask.return_value = "LIM"

        action = self.load.get_overcurrent_protection_action()

        self.mock_adapter.ask.assert_called_with("SOURce:CURRent:PROTection:ACTion?")
        self.assertEqual(action, "LIM")

    def test_get_overcurrent_protection_action_off(self):
        """Test querying OCP action when set to OFF."""
        self.mock_adapter.ask.return_value = "OFF"

        action = self.load.get_overcurrent_protection_action()

        self.assertEqual(action, "OFF")

    # --- Undervoltage Protection (UVP) Tests ---

    def test_set_undervoltage_protection_valid(self):
        """Test setting valid UVP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_undervoltage_protection(10.5, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 10.5")

    def test_set_undervoltage_protection_max(self):
        """Test setting maximum UVP threshold."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_undervoltage_protection(150.0, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 150.0")

    def test_set_undervoltage_protection_zero(self):
        """Test setting UVP threshold to zero."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        self.load.set_undervoltage_protection(0.0, verify=False)

        self.mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 0.0")

    def test_set_undervoltage_protection_out_of_range_high(self):
        """Test error when UVP threshold too high."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_undervoltage_protection(200.0)

        self.assertIn("out of range", str(cm.exception))

    def test_set_undervoltage_protection_negative(self):
        """Test error when UVP threshold negative."""
        with self.assertRaises(ValueError) as cm:
            self.load.set_undervoltage_protection(-5.0)

        self.assertIn("out of range", str(cm.exception))

    def test_get_undervoltage_protection(self):
        """Test querying UVP threshold."""
        self.mock_adapter.ask.return_value = "10.500"

        uvp = self.load.get_undervoltage_protection()

        self.mock_adapter.ask.assert_called_with("SOURce:VOLTage:PROTection:UNDer?")
        self.assertAlmostEqual(uvp, 10.5)


if __name__ == '__main__':
    unittest.main()
