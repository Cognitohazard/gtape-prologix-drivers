"""Unit tests for AgilentE3631A power supply driver."""

import pytest
from unittest.mock import patch
from gtape_prologix_drivers.instruments.agilent_e3631a import AgilentE3631A


class TestAgilentE3631A:
    """Test cases for AgilentE3631A power supply driver."""

    def test_initialization(self, psu, mock_adapter):
        """Test PSU initialization."""
        assert psu.adapter == mock_adapter
        assert psu._current_channel is None

    def test_channel_constants(self):
        """Test channel constant definitions."""
        assert AgilentE3631A.P6V == 1
        assert AgilentE3631A.P25V == 2
        assert AgilentE3631A.N25V == 3

    def test_channel_specs(self):
        """Test channel specifications."""
        assert AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.P6V] == (6.0, 5.0)
        assert AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.P25V] == (25.0, 1.0)
        assert AgilentE3631A.CHANNEL_SPECS[AgilentE3631A.N25V] == (25.0, 1.0)

    def test_reset(self, psu, mock_adapter):
        """Test PSU reset."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.reset()

        mock_adapter.write.assert_called_with("*RST")
        mock_adapter.ask.assert_called_with("SYST:ERR?")
        assert psu._current_channel is None

    def test_select_channel_p6v(self, psu, mock_adapter):
        """Test selecting P6V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.select_channel(AgilentE3631A.P6V)

        mock_adapter.write.assert_called_with("INST:NSEL 1")
        assert psu._current_channel == AgilentE3631A.P6V

    def test_select_channel_p25v(self, psu, mock_adapter):
        """Test selecting P25V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.select_channel(AgilentE3631A.P25V)

        mock_adapter.write.assert_called_with("INST:NSEL 2")
        assert psu._current_channel == AgilentE3631A.P25V

    def test_select_channel_n25v(self, psu, mock_adapter):
        """Test selecting N25V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.select_channel(AgilentE3631A.N25V)

        mock_adapter.write.assert_called_with("INST:NSEL 3")
        assert psu._current_channel == AgilentE3631A.N25V

    def test_select_channel_invalid(self, psu):
        """Test error when selecting invalid channel."""
        with pytest.raises(ValueError) as excinfo:
            psu.select_channel(5)

        assert "Invalid channel" in str(excinfo.value)

    def test_set_voltage_p6v_valid(self, psu, mock_adapter):
        """Test setting valid voltage on P6V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'
        psu._current_channel = AgilentE3631A.P6V

        psu.set_voltage(5.0)

        mock_adapter.write.assert_called_with("VOLT 5.0")

    def test_set_voltage_p25v_valid(self, psu, mock_adapter):
        """Test setting valid voltage on P25V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'
        psu._current_channel = AgilentE3631A.P25V

        psu.set_voltage(20.0)

        mock_adapter.write.assert_called_with("VOLT 20.0")

    def test_set_voltage_n25v_valid(self, psu, mock_adapter):
        """Test setting valid negative voltage on N25V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'
        psu._current_channel = AgilentE3631A.N25V

        psu.set_voltage(-15.0)

        mock_adapter.write.assert_called_with("VOLT -15.0")

    def test_set_voltage_no_channel_selected(self, psu):
        """Test error when setting voltage without selecting channel."""
        with pytest.raises(ValueError) as excinfo:
            psu.set_voltage(5.0)

        assert "No channel selected" in str(excinfo.value)

    def test_set_voltage_p6v_out_of_range_high(self, psu):
        """Test error when P6V voltage too high."""
        psu._current_channel = AgilentE3631A.P6V

        with pytest.raises(ValueError) as excinfo:
            psu.set_voltage(7.0)

        assert "out of range" in str(excinfo.value)

    def test_set_voltage_p6v_out_of_range_negative(self, psu):
        """Test error when P6V voltage negative."""
        psu._current_channel = AgilentE3631A.P6V

        with pytest.raises(ValueError) as excinfo:
            psu.set_voltage(-1.0)

        assert "out of range" in str(excinfo.value)

    def test_set_voltage_n25v_positive_error(self, psu):
        """Test error when N25V channel given positive voltage."""
        psu._current_channel = AgilentE3631A.N25V

        with pytest.raises(ValueError) as excinfo:
            psu.set_voltage(10.0)

        assert "negative voltage" in str(excinfo.value)

    def test_set_voltage_n25v_out_of_range(self, psu):
        """Test error when N25V voltage magnitude too large."""
        psu._current_channel = AgilentE3631A.N25V

        with pytest.raises(ValueError) as excinfo:
            psu.set_voltage(-30.0)

        assert "exceeds N25V range" in str(excinfo.value)

    def test_set_current_limit_p6v_valid(self, psu, mock_adapter):
        """Test setting valid current limit on P6V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'
        psu._current_channel = AgilentE3631A.P6V

        psu.set_current_limit(3.0)

        mock_adapter.write.assert_called_with("CURR 3.0")

    def test_set_current_limit_p25v_valid(self, psu, mock_adapter):
        """Test setting valid current limit on P25V channel."""
        mock_adapter.ask.return_value = '+0,"No error"'
        psu._current_channel = AgilentE3631A.P25V

        psu.set_current_limit(0.5)

        mock_adapter.write.assert_called_with("CURR 0.5")

    def test_set_current_limit_no_channel_selected(self, psu):
        """Test error when setting current without selecting channel."""
        with pytest.raises(ValueError) as excinfo:
            psu.set_current_limit(1.0)

        assert "No channel selected" in str(excinfo.value)

    def test_set_current_limit_p6v_out_of_range(self, psu):
        """Test error when P6V current limit too high."""
        psu._current_channel = AgilentE3631A.P6V

        with pytest.raises(ValueError) as excinfo:
            psu.set_current_limit(6.0)

        assert "out of range" in str(excinfo.value)

    def test_set_current_limit_p25v_out_of_range(self, psu):
        """Test error when P25V current limit too high."""
        psu._current_channel = AgilentE3631A.P25V

        with pytest.raises(ValueError) as excinfo:
            psu.set_current_limit(2.0)

        assert "out of range" in str(excinfo.value)

    def test_set_current_limit_negative(self, psu):
        """Test error when current limit negative."""
        psu._current_channel = AgilentE3631A.P6V

        with pytest.raises(ValueError) as excinfo:
            psu.set_current_limit(-1.0)

        assert "out of range" in str(excinfo.value)

    def test_enable_output_true(self, psu, mock_adapter):
        """Test enabling output."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.enable_output(True)

        mock_adapter.write.assert_called_with("OUTP 1")

    def test_enable_output_false(self, psu, mock_adapter):
        """Test disabling output."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.enable_output(False)

        mock_adapter.write.assert_called_with("OUTP 0")

    def test_enable_output_default(self, psu, mock_adapter):
        """Test enable_output default value (True)."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.enable_output()

        mock_adapter.write.assert_called_with("OUTP 1")

    def test_measure_voltage_with_channel(self, psu, mock_adapter):
        """Test measuring voltage with explicit channel."""
        mock_adapter.ask.side_effect = ['+0,"No error"', "+5.013"]

        voltage = psu.measure_voltage(channel=AgilentE3631A.P6V)

        assert psu._current_channel == AgilentE3631A.P6V
        assert voltage == pytest.approx(5.013)

    def test_measure_voltage_no_channel_error(self, psu):
        """Test error when measuring voltage without channel selected."""
        with pytest.raises(ValueError) as excinfo:
            psu.measure_voltage()

        assert "No channel selected" in str(excinfo.value)

    def test_measure_voltage_current_channel(self, psu, mock_adapter):
        """Test measuring voltage with current channel."""
        mock_adapter.ask.return_value = "+5.013"
        psu._current_channel = AgilentE3631A.P6V

        voltage = psu.measure_voltage()

        mock_adapter.ask.assert_called_with("MEAS:VOLT?")
        assert voltage == pytest.approx(5.013)

    def test_measure_current_with_channel(self, psu, mock_adapter):
        """Test measuring current with explicit channel."""
        mock_adapter.ask.side_effect = ['+0,"No error"', "+0.1234"]

        current = psu.measure_current(channel=AgilentE3631A.P6V)

        assert psu._current_channel == AgilentE3631A.P6V
        assert current == pytest.approx(0.1234)

    def test_measure_current_no_channel_error(self, psu):
        """Test error when measuring current without channel selected."""
        with pytest.raises(ValueError) as excinfo:
            psu.measure_current()

        assert "No channel selected" in str(excinfo.value)

    def test_measure_current_current_channel(self, psu, mock_adapter):
        """Test measuring current with current channel."""
        mock_adapter.ask.return_value = "+0.1234"
        psu._current_channel = AgilentE3631A.P6V

        current = psu.measure_current()

        mock_adapter.ask.assert_called_with("MEAS:CURR?")
        assert current == pytest.approx(0.1234)

    def test_check_errors_no_error(self, psu, mock_adapter):
        """Test error checking with no errors."""
        mock_adapter.ask.return_value = '+0,"No error"'

        result = psu.check_errors()

        assert result == '+0,"No error"'
        mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_check_errors_alternative_format(self, psu, mock_adapter):
        """Test error checking with alternative no-error format."""
        mock_adapter.ask.return_value = '0,"No error"'

        result = psu.check_errors()

        assert result == '0,"No error"'

    def test_check_errors_with_error(self, psu, mock_adapter):
        """Test error checking when error exists."""
        mock_adapter.ask.return_value = '-113,"Undefined header"'

        with patch('builtins.print') as mock_print:
            result = psu.check_errors()

        assert result == '-113,"Undefined header"'
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error" in str(call) for call in print_calls)

    def test_configure_output_complete(self, psu, mock_adapter):
        """Test complete configuration workflow."""
        mock_adapter.ask.return_value = '+0,"No error"'

        psu.configure_output(AgilentE3631A.P25V, 12.0, 0.5)

        write_calls = [str(call) for call in mock_adapter.write.call_args_list]
        assert any("INST:NSEL 2" in str(call) for call in write_calls)
        assert any("VOLT 12.0" in str(call) for call in write_calls)
        assert any("CURR 0.5" in str(call) for call in write_calls)
        assert psu._current_channel == AgilentE3631A.P25V
