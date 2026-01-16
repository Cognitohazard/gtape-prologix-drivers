"""Unit tests for PLZ164W electronic load driver."""

import pytest
from unittest.mock import patch
from gtape_prologix_drivers.instruments.plz164w import PLZ164W


class TestPLZ164W:
    """Test cases for PLZ164W electronic load driver."""

    def test_initialization(self, load, mock_adapter):
        """Test load initialization."""
        assert load.adapter == mock_adapter
        assert load._current_mode is None

    def test_mode_constants(self):
        """Test mode constant definitions."""
        assert PLZ164W.MODE_CC == "CURR"
        assert PLZ164W.MODE_CV == "VOLT"
        assert PLZ164W.MODE_CR == "RES"
        assert PLZ164W.MODE_CP == "POW"
        assert PLZ164W.MODE_CCCV == "CCCV"
        assert PLZ164W.MODE_CRCV == "CRCV"

    def test_range_constants(self):
        """Test range constant definitions."""
        assert PLZ164W.CURR_RANGE_LOW == "LOW"
        assert PLZ164W.CURR_RANGE_MED == "MEDIUM"
        assert PLZ164W.CURR_RANGE_HIGH == "HIGH"
        assert PLZ164W.VOLT_RANGE_LOW == "LOW"
        assert PLZ164W.VOLT_RANGE_HIGH == "HIGH"

    def test_specification_constants(self):
        """Test specification constants."""
        assert PLZ164W.VOLTAGE_MIN == 1.5
        assert PLZ164W.VOLTAGE_MAX == 150.0
        assert PLZ164W.CURRENT_MAX == 33.0
        assert PLZ164W.POWER_MAX == 165.0
        assert PLZ164W.RESISTANCE_MIN == 0.5
        assert PLZ164W.RESISTANCE_MAX == 6000.0

    def test_reset(self, load, mock_adapter):
        """Test load reset."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.reset()

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("*RST" in str(call) for call in write_calls)
        mock_adapter.ask.assert_called_with("SYSTem:ERRor?")
        assert load._current_mode is None

    def test_get_identification(self, load, mock_adapter):
        """Test querying instrument identification."""
        mock_adapter.ask.return_value = "KIKUSUI,PLZ164W,12345,1.00"

        idn = load.get_identification()

        mock_adapter.ask.assert_called_with("*IDN?")
        assert idn == "KIKUSUI,PLZ164W,12345,1.00"

    def test_set_mode_cc(self, load, mock_adapter):
        """Test setting CC mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_mode(PLZ164W.MODE_CC)

        mock_adapter.write.assert_called_with("SOURce:FUNCtion CURR")
        assert load._current_mode == PLZ164W.MODE_CC

    def test_set_mode_cv(self, load, mock_adapter):
        """Test setting CV mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_mode(PLZ164W.MODE_CV)

        mock_adapter.write.assert_called_with("SOURce:FUNCtion VOLT")
        assert load._current_mode == PLZ164W.MODE_CV

    def test_set_mode_cr(self, load, mock_adapter):
        """Test setting CR mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_mode(PLZ164W.MODE_CR)

        mock_adapter.write.assert_called_with("SOURce:FUNCtion RES")
        assert load._current_mode == PLZ164W.MODE_CR

    def test_set_mode_cp(self, load, mock_adapter):
        """Test setting CP mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_mode(PLZ164W.MODE_CP)

        mock_adapter.write.assert_called_with("SOURce:FUNCtion POW")
        assert load._current_mode == PLZ164W.MODE_CP

    def test_set_mode_invalid(self, load):
        """Test error when setting invalid mode."""
        with pytest.raises(ValueError) as excinfo:
            load.set_mode("INVALID")

        assert "Invalid mode" in str(excinfo.value)

    def test_get_mode(self, load, mock_adapter):
        """Test querying current mode."""
        mock_adapter.ask.return_value = "CURR"

        mode = load.get_mode()

        mock_adapter.ask.assert_called_with("SOURce:FUNCtion?")
        assert mode == "CURR"

    def test_set_current_valid(self, load, mock_adapter):
        """Test setting valid current."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_current(5.0)

        mock_adapter.write.assert_called_with("SOURce:CURRent 5.0")

    def test_set_current_max(self, load, mock_adapter):
        """Test setting maximum current."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_current(33.0)

        mock_adapter.write.assert_called_with("SOURce:CURRent 33.0")

    def test_set_current_out_of_range_high(self, load):
        """Test error when current too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_current(40.0)

        assert "out of range" in str(excinfo.value)

    def test_set_current_negative(self, load):
        """Test error when current negative."""
        with pytest.raises(ValueError) as excinfo:
            load.set_current(-1.0)

        assert "out of range" in str(excinfo.value)

    def test_get_current(self, load, mock_adapter):
        """Test querying current setting."""
        mock_adapter.ask.return_value = "2.500"

        current = load.get_current()

        mock_adapter.ask.assert_called_with("SOURce:CURRent?")
        assert current == pytest.approx(2.5)

    def test_set_voltage_valid(self, load, mock_adapter):
        """Test setting valid voltage."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_voltage(12.0)

        mock_adapter.write.assert_called_with("SOURce:VOLTage 12.0")

    def test_set_voltage_max(self, load, mock_adapter):
        """Test setting maximum voltage."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_voltage(150.0)

        mock_adapter.write.assert_called_with("SOURce:VOLTage 150.0")

    def test_set_voltage_out_of_range_high(self, load):
        """Test error when voltage too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_voltage(200.0)

        assert "out of range" in str(excinfo.value)

    def test_set_voltage_out_of_range_low(self, load):
        """Test error when voltage too low."""
        with pytest.raises(ValueError) as excinfo:
            load.set_voltage(1.0)

        assert "out of range" in str(excinfo.value)

    def test_get_voltage(self, load, mock_adapter):
        """Test querying voltage setting."""
        mock_adapter.ask.return_value = "12.000"

        voltage = load.get_voltage()

        mock_adapter.ask.assert_called_with("SOURce:VOLTage?")
        assert voltage == pytest.approx(12.0)

    def test_set_power_valid(self, load, mock_adapter):
        """Test setting valid power."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_power(50.0)

        mock_adapter.write.assert_called_with("SOURce:POWer 50.0")

    def test_set_power_max(self, load, mock_adapter):
        """Test setting maximum power."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_power(165.0)

        mock_adapter.write.assert_called_with("SOURce:POWer 165.0")

    def test_set_power_out_of_range_high(self, load):
        """Test error when power too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_power(200.0)

        assert "out of range" in str(excinfo.value)

    def test_set_power_negative(self, load):
        """Test error when power negative."""
        with pytest.raises(ValueError) as excinfo:
            load.set_power(-10.0)

        assert "out of range" in str(excinfo.value)

    def test_get_power(self, load, mock_adapter):
        """Test querying power setting."""
        mock_adapter.ask.return_value = "50.000"

        power = load.get_power()

        mock_adapter.ask.assert_called_with("SOURce:POWer?")
        assert power == pytest.approx(50.0)

    def test_set_resistance_valid(self, load, mock_adapter):
        """Test setting valid resistance."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_resistance(100.0)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:CONDuctance 0.01" in str(call) for call in write_calls)

    def test_set_resistance_min(self, load, mock_adapter):
        """Test setting minimum resistance."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_resistance(0.5)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:CONDuctance 2.0" in str(call) for call in write_calls)

    def test_set_resistance_out_of_range_high(self, load):
        """Test error when resistance too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_resistance(7000.0)

        assert "out of range" in str(excinfo.value)

    def test_set_resistance_out_of_range_low(self, load):
        """Test error when resistance too low."""
        with pytest.raises(ValueError) as excinfo:
            load.set_resistance(0.1)

        assert "out of range" in str(excinfo.value)

    def test_get_resistance(self, load, mock_adapter):
        """Test querying resistance setting."""
        mock_adapter.ask.return_value = "0.01"

        resistance = load.get_resistance()

        mock_adapter.ask.assert_called_with("SOURce:CONDuctance?")
        assert resistance == pytest.approx(100.0)

    def test_get_resistance_zero_conductance(self, load, mock_adapter):
        """Test querying resistance when conductance is zero."""
        mock_adapter.ask.return_value = "0.0"

        resistance = load.get_resistance()

        assert resistance == float('inf')

    def test_set_current_range_low(self, load, mock_adapter):
        """Test setting low current range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_current_range(PLZ164W.CURR_RANGE_LOW)

        mock_adapter.write.assert_called_with("SOURce:CURRent:RANGe LOW")

    def test_set_current_range_high(self, load, mock_adapter):
        """Test setting high current range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_current_range(PLZ164W.CURR_RANGE_HIGH)

        mock_adapter.write.assert_called_with("SOURce:CURRent:RANGe HIGH")

    def test_set_current_range_invalid(self, load):
        """Test error when setting invalid current range."""
        with pytest.raises(ValueError) as excinfo:
            load.set_current_range("INVALID")

        assert "Invalid current range" in str(excinfo.value)

    def test_set_voltage_range_low(self, load, mock_adapter):
        """Test setting low voltage range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_voltage_range(PLZ164W.VOLT_RANGE_LOW)

        mock_adapter.write.assert_called_with("SOURce:VOLTage:RANGe LOW")

    def test_set_voltage_range_high(self, load, mock_adapter):
        """Test setting high voltage range."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_voltage_range(PLZ164W.VOLT_RANGE_HIGH)

        mock_adapter.write.assert_called_with("SOURce:VOLTage:RANGe HIGH")

    def test_set_voltage_range_invalid(self, load):
        """Test error when setting invalid voltage range."""
        with pytest.raises(ValueError) as excinfo:
            load.set_voltage_range("INVALID")

        assert "Invalid voltage range" in str(excinfo.value)

    def test_enable_input_true(self, load, mock_adapter):
        """Test enabling input."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.enable_input(True)

        mock_adapter.write.assert_called_with("INPut 1")
        # Verify check_errors was called (via mock_adapter.ask)
        mock_adapter.ask.assert_called()

    def test_enable_input_false(self, load, mock_adapter):
        """Test disabling input."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.enable_input(False)

        mock_adapter.write.assert_called_with("INPut 0")

    def test_enable_input_default(self, load, mock_adapter):
        """Test enable_input default value (True)."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.enable_input()

        mock_adapter.write.assert_called_with("INPut 1")

    def test_get_input_state_enabled(self, load, mock_adapter):
        """Test querying input state when enabled."""
        mock_adapter.ask.return_value = "1"

        state = load.get_input_state()

        mock_adapter.ask.assert_called_with("INPut?")
        assert state is True

    def test_get_input_state_disabled(self, load, mock_adapter):
        """Test querying input state when disabled."""
        mock_adapter.ask.return_value = "0"

        state = load.get_input_state()

        mock_adapter.ask.assert_called_with("INPut?")
        assert state is False

    def test_set_short_mode_enable(self, load, mock_adapter):
        """Test enabling short mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_short_mode(True)

        mock_adapter.write.assert_called_with("INPut:SHORt 1")

    def test_set_short_mode_disable(self, load, mock_adapter):
        """Test disabling short mode."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_short_mode(False)

        mock_adapter.write.assert_called_with("INPut:SHORt 0")

    def test_set_short_mode_default(self, load, mock_adapter):
        """Test set_short_mode default value (False)."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_short_mode()

        mock_adapter.write.assert_called_with("INPut:SHORt 0")

    def test_measure_voltage(self, load, mock_adapter):
        """Test measuring voltage."""
        mock_adapter.ask.return_value = "12.345"

        voltage = load.measure_voltage()

        mock_adapter.ask.assert_called_with("MEASure:VOLTage?")
        assert voltage == pytest.approx(12.345)

    def test_measure_current(self, load, mock_adapter):
        """Test measuring current."""
        mock_adapter.ask.return_value = "2.567"

        current = load.measure_current()

        mock_adapter.ask.assert_called_with("MEASure:CURRent?")
        assert current == pytest.approx(2.567)

    def test_measure_power(self, load, mock_adapter):
        """Test measuring power."""
        mock_adapter.ask.return_value = "31.695"

        power = load.measure_power()

        mock_adapter.ask.assert_called_with("MEASure:POWer?")
        assert power == pytest.approx(31.695)

    def test_check_errors_no_error(self, load, mock_adapter):
        """Test error checking with no errors."""
        mock_adapter.ask.return_value = '+0,"No error"'

        result = load.check_errors()

        assert result == '+0,"No error"'
        mock_adapter.ask.assert_called_with("SYSTem:ERRor?")

    def test_check_errors_alternative_format(self, load, mock_adapter):
        """Test error checking with alternative no-error format."""
        mock_adapter.ask.return_value = '0,"No error"'

        result = load.check_errors()

        assert result == '0,"No error"'

    def test_check_errors_with_error(self, load, mock_adapter):
        """Test error checking when error exists."""
        mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = load.check_errors()

        assert result == '-113,"Undefined header"'
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error" in str(call) for call in print_calls)

    def test_configure_cc_mode(self, load, mock_adapter):
        """Test complete CC mode configuration."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.configure_cc_mode(5.0, PLZ164W.CURR_RANGE_HIGH)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:FUNCtion CURR" in str(call) for call in write_calls)
        assert any("SOURce:CURRent:RANGe HIGH" in str(call) for call in write_calls)
        assert any("SOURce:CURRent 5.0" in str(call) for call in write_calls)
        assert load._current_mode == PLZ164W.MODE_CC

    def test_configure_cc_mode_no_range(self, load, mock_adapter):
        """Test CC mode configuration without range setting."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.configure_cc_mode(5.0)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:CURRent 5.0" in str(call) for call in write_calls)
        assert not any("RANG" in str(call) for call in write_calls)

    def test_configure_cv_mode(self, load, mock_adapter):
        """Test complete CV mode configuration."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.configure_cv_mode(12.0, PLZ164W.VOLT_RANGE_HIGH)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:FUNCtion VOLT" in str(call) for call in write_calls)
        assert any("SOURce:VOLTage:RANGe HIGH" in str(call) for call in write_calls)
        assert any("SOURce:VOLTage 12.0" in str(call) for call in write_calls)
        assert load._current_mode == PLZ164W.MODE_CV

    def test_configure_cr_mode(self, load, mock_adapter):
        """Test complete CR mode configuration."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.configure_cr_mode(100.0)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:FUNCtion RES" in str(call) for call in write_calls)
        assert any("SOURce:CONDuctance" in str(call) for call in write_calls)
        assert load._current_mode == PLZ164W.MODE_CR

    def test_configure_cp_mode(self, load, mock_adapter):
        """Test complete CP mode configuration."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.configure_cp_mode(50.0)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("SOURce:FUNCtion POW" in str(call) for call in write_calls)
        assert any("SOURce:POWer 50.0" in str(call) for call in write_calls)
        assert load._current_mode == PLZ164W.MODE_CP

    # --- Overpower Protection (OPP) Tests ---

    def test_opp_action_constants(self):
        """Test OPP action constant definitions."""
        assert PLZ164W.OPP_ACTION_LIMIT == "LIM"

    def test_opp_max_constant(self):
        """Test OPP maximum constant."""
        assert PLZ164W.OPP_MAX == 181.5

    def test_set_overpower_protection_valid(self, load, mock_adapter):
        """Test setting valid OPP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overpower_protection(100.0, verify=False)

        mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 100.0")

    def test_set_overpower_protection_max(self, load, mock_adapter):
        """Test setting maximum OPP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overpower_protection(181.5, verify=False)

        mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 181.5")

    def test_set_overpower_protection_zero(self, load, mock_adapter):
        """Test setting OPP threshold to zero."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overpower_protection(0.0, verify=False)

        mock_adapter.write.assert_called_with("SOURce:POWer:PROTection 0.0")

    def test_set_overpower_protection_out_of_range_high(self, load):
        """Test error when OPP threshold too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overpower_protection(200.0)

        assert "out of range" in str(excinfo.value)

    def test_set_overpower_protection_negative(self, load):
        """Test error when OPP threshold negative."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overpower_protection(-10.0)

        assert "out of range" in str(excinfo.value)

    def test_get_overpower_protection(self, load, mock_adapter):
        """Test querying OPP threshold."""
        mock_adapter.ask.return_value = "150.000"

        opp = load.get_overpower_protection()

        mock_adapter.ask.assert_called_with("SOURce:POWer:PROTection?")
        assert opp == pytest.approx(150.0)

    def test_set_overpower_protection_action_limit(self, load, mock_adapter):
        """Test setting OPP action to LIMIT."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overpower_protection_action(PLZ164W.OPP_ACTION_LIMIT)

        mock_adapter.write.assert_called_with("SOURce:POWer:PROTection:ACTion LIM")

    def test_set_overpower_protection_action_invalid(self, load):
        """Test error when setting non-LIM OPP action."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overpower_protection_action("OFF")

        assert "Only 'LIM' supported via SCPI" in str(excinfo.value)

    def test_get_overpower_protection_action(self, load, mock_adapter):
        """Test querying OPP action setting."""
        mock_adapter.ask.return_value = "LIM"

        action = load.get_overpower_protection_action()

        mock_adapter.ask.assert_called_with("SOURce:POWer:PROTection:ACTion?")
        assert action == "LIM"

    def test_get_overpower_protection_action_off(self, load, mock_adapter):
        """Test querying OPP action when set to OFF."""
        mock_adapter.ask.return_value = "OFF"

        action = load.get_overpower_protection_action()

        assert action == "OFF"

    # --- Overcurrent Protection (OCP) Tests ---

    def test_protection_action_constants(self):
        """Test protection action constant definitions."""
        assert PLZ164W.PROT_ACTION_LIMIT == "LIM"

    def test_ocp_max_constant(self):
        """Test OCP maximum constant."""
        assert PLZ164W.OCP_MAX == 36.29

    def test_set_overcurrent_protection_valid(self, load, mock_adapter):
        """Test setting valid OCP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overcurrent_protection(10.0)

        mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 10.0")

    def test_set_overcurrent_protection_max(self, load, mock_adapter):
        """Test setting maximum OCP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overcurrent_protection(36.29)

        mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 36.29")

    def test_set_overcurrent_protection_zero(self, load, mock_adapter):
        """Test setting OCP threshold to zero."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overcurrent_protection(0.0)

        mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection 0.0")

    def test_set_overcurrent_protection_out_of_range_high(self, load):
        """Test error when OCP threshold too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overcurrent_protection(40.0)

        assert "out of range" in str(excinfo.value)

    def test_set_overcurrent_protection_negative(self, load):
        """Test error when OCP threshold negative."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overcurrent_protection(-5.0)

        assert "out of range" in str(excinfo.value)

    def test_get_overcurrent_protection(self, load, mock_adapter):
        """Test querying OCP threshold."""
        mock_adapter.ask.return_value = "10.000"

        ocp = load.get_overcurrent_protection()

        mock_adapter.ask.assert_called_with("SOURce:CURRent:PROTection?")
        assert ocp == pytest.approx(10.0)

    def test_set_overcurrent_protection_action_limit(self, load, mock_adapter):
        """Test setting OCP action to LIMIT."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_overcurrent_protection_action(PLZ164W.PROT_ACTION_LIMIT)

        mock_adapter.write.assert_called_with("SOURce:CURRent:PROTection:ACTion LIM")

    def test_set_overcurrent_protection_action_invalid(self, load):
        """Test error when setting non-LIM OCP action."""
        with pytest.raises(ValueError) as excinfo:
            load.set_overcurrent_protection_action("OFF")

        assert "Only 'LIM' supported via SCPI" in str(excinfo.value)

    def test_get_overcurrent_protection_action(self, load, mock_adapter):
        """Test querying OCP action setting."""
        mock_adapter.ask.return_value = "LIM"

        action = load.get_overcurrent_protection_action()

        mock_adapter.ask.assert_called_with("SOURce:CURRent:PROTection:ACTion?")
        assert action == "LIM"

    def test_get_overcurrent_protection_action_off(self, load, mock_adapter):
        """Test querying OCP action when set to OFF."""
        mock_adapter.ask.return_value = "OFF"

        action = load.get_overcurrent_protection_action()

        assert action == "OFF"

    # --- Undervoltage Protection (UVP) Tests ---

    def test_set_undervoltage_protection_valid(self, load, mock_adapter):
        """Test setting valid UVP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_undervoltage_protection(10.5, verify=False)

        mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 10.5")

    def test_set_undervoltage_protection_max(self, load, mock_adapter):
        """Test setting maximum UVP threshold."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_undervoltage_protection(150.0, verify=False)

        mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 150.0")

    def test_set_undervoltage_protection_zero(self, load, mock_adapter):
        """Test setting UVP threshold to zero."""
        mock_adapter.ask.return_value = '+0,"No error"'

        load.set_undervoltage_protection(0.0, verify=False)

        mock_adapter.write.assert_called_with("SOURce:VOLTage:PROTection:UNDer 0.0")

    def test_set_undervoltage_protection_out_of_range_high(self, load):
        """Test error when UVP threshold too high."""
        with pytest.raises(ValueError) as excinfo:
            load.set_undervoltage_protection(200.0)

        assert "out of range" in str(excinfo.value)

    def test_set_undervoltage_protection_negative(self, load):
        """Test error when UVP threshold negative."""
        with pytest.raises(ValueError) as excinfo:
            load.set_undervoltage_protection(-5.0)

        assert "out of range" in str(excinfo.value)

    def test_get_undervoltage_protection(self, load, mock_adapter):
        """Test querying UVP threshold."""
        mock_adapter.ask.return_value = "10.500"

        uvp = load.get_undervoltage_protection()

        mock_adapter.ask.assert_called_with("SOURce:VOLTage:PROTection:UNDer?")
        assert uvp == pytest.approx(10.5)
