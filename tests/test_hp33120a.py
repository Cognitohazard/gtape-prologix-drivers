"""Unit tests for HP33120A AWG driver."""

import pytest
from unittest.mock import Mock, call
import numpy as np
from gtape_prologix_drivers.instruments.hp33120a import HP33120A


class TestHP33120AConstants:
    """Test class constants."""

    def test_dac_range_is_signed(self):
        """DAC range should be signed bipolar -2047 to +2047."""
        assert HP33120A.DAC_MIN == -2047
        assert HP33120A.DAC_MAX == 2047

    def test_point_limits(self):
        """Waveform point limits."""
        assert HP33120A.MIN_POINTS == 8
        assert HP33120A.MAX_POINTS == 16000

    def test_shape_constants(self):
        """Function shape constants match SCPI commands."""
        assert HP33120A.SHAPE_SINE == "SIN"
        assert HP33120A.SHAPE_SQUARE == "SQU"
        assert HP33120A.SHAPE_TRIANGLE == "TRI"
        assert HP33120A.SHAPE_RAMP == "RAMP"
        assert HP33120A.SHAPE_NOISE == "NOIS"
        assert HP33120A.SHAPE_DC == "DC"
        assert HP33120A.SHAPE_USER == "USER"

    def test_unit_constants(self):
        """Voltage unit constants."""
        assert HP33120A.UNIT_VPP == "VPP"
        assert HP33120A.UNIT_VRMS == "VRMS"
        assert HP33120A.UNIT_DBM == "DBM"

    def test_load_constants(self):
        """Load impedance constants."""
        assert HP33120A.LOAD_50_OHM == 50
        assert HP33120A.LOAD_HIGH_Z == "INF"

    def test_trigger_constants(self):
        """Trigger source constants."""
        assert HP33120A.TRIGGER_IMMEDIATE == "IMM"
        assert HP33120A.TRIGGER_EXTERNAL == "EXT"
        assert HP33120A.TRIGGER_BUS == "BUS"

    def test_sweep_constants(self):
        """Sweep spacing constants."""
        assert HP33120A.SWEEP_LINEAR == "LIN"
        assert HP33120A.SWEEP_LOGARITHMIC == "LOG"


class TestHP33120AInit:
    """Test initialization."""

    def test_initialization(self, awg):
        """AWG stores adapter reference."""
        assert awg.adapter is not None


class TestHP33120ASystemMethods:
    """Test system methods."""

    def test_reset(self, awg):
        """Reset sends *RST and waits for completion."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.reset()
        awg.adapter.write.assert_any_call("*RST")
        awg.adapter.ask.assert_any_call("*OPC?")

    def test_get_identification(self, awg):
        """get_identification queries *IDN?."""
        awg.adapter.ask.return_value = "HEWLETT-PACKARD,33120A,0,1.0"
        result = awg.get_identification()
        awg.adapter.ask.assert_called_with("*IDN?")
        assert "33120A" in result

    def test_self_test(self, awg):
        """self_test returns integer result."""
        awg.adapter.ask.return_value = "0"
        result = awg.self_test()
        awg.adapter.ask.assert_called_with("*TST?")
        assert result == 0

    def test_get_version(self, awg):
        """get_version queries SCPI version."""
        awg.adapter.ask.return_value = "1994.0"
        result = awg.get_version()
        awg.adapter.ask.assert_called_with("SYST:VERS?")
        assert "1994" in result

    def test_beep(self, awg):
        """beep sends SYST:BEEP command."""
        awg.beep()
        awg.adapter.write.assert_called_with("SYST:BEEP")

    def test_save_state(self, awg):
        """save_state sends *SAV command."""
        awg.adapter.ask.return_value = "1"
        awg.save_state(2)
        awg.adapter.write.assert_any_call("*SAV 2")

    def test_save_state_invalid_location(self, awg):
        """save_state rejects invalid locations."""
        with pytest.raises(ValueError, match="0-3"):
            awg.save_state(5)

    def test_recall_state(self, awg):
        """recall_state sends *RCL command."""
        awg.adapter.ask.return_value = "1"
        awg.recall_state(1)
        awg.adapter.write.assert_any_call("*RCL 1")

    def test_delete_state(self, awg):
        """delete_state sends MEM:STAT:DEL command."""
        awg.delete_state(3)
        awg.adapter.write.assert_called_with("MEM:STAT:DEL 3")

    def test_check_errors_no_error(self, awg):
        """check_errors returns error string."""
        awg.adapter.ask.return_value = '+0,"No error"'
        result = awg.check_errors()
        assert result == '+0,"No error"'


class TestHP33120ADisplayMethods:
    """Test display methods."""

    def test_set_display_on(self, awg):
        """set_display(True) sends DISP ON."""
        awg.set_display(True)
        awg.adapter.write.assert_called_with("DISP ON")

    def test_set_display_off(self, awg):
        """set_display(False) sends DISP OFF."""
        awg.set_display(False)
        awg.adapter.write.assert_called_with("DISP OFF")

    def test_get_display(self, awg):
        """get_display queries display state."""
        awg.adapter.ask.return_value = "1"
        result = awg.get_display()
        awg.adapter.ask.assert_called_with("DISP?")
        assert result is True

    def test_set_display_text(self, awg):
        """set_display_text sends text to display."""
        awg.set_display_text("TESTING")
        awg.adapter.write.assert_called_with("DISP:TEXT 'TESTING'")

    def test_get_display_text(self, awg):
        """get_display_text queries display text."""
        awg.adapter.ask.return_value = '"HELLO"'
        result = awg.get_display_text()
        awg.adapter.ask.assert_called_with("DISP:TEXT?")

    def test_clear_display_text(self, awg):
        """clear_display_text sends clear command."""
        awg.clear_display_text()
        awg.adapter.write.assert_called_with("DISP:TEXT:CLE")


class TestHP33120AOutputParameters:
    """Test output parameter getters/setters."""

    def test_set_frequency(self, awg):
        """set_frequency sends FREQ command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_frequency(1000)
        awg.adapter.write.assert_called_with("FREQ 1000")

    def test_get_frequency(self, awg):
        """get_frequency queries frequency."""
        awg.adapter.ask.return_value = "1000.0"
        result = awg.get_frequency()
        awg.adapter.ask.assert_called_with("FREQ?")
        assert result == 1000.0

    def test_set_amplitude(self, awg):
        """set_amplitude sends VOLT command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_amplitude(2.5)
        awg.adapter.write.assert_called_with("VOLT 2.5")

    def test_get_amplitude(self, awg):
        """get_amplitude queries amplitude."""
        awg.adapter.ask.return_value = "2.5"
        result = awg.get_amplitude()
        awg.adapter.ask.assert_called_with("VOLT?")
        assert result == 2.5

    def test_set_offset(self, awg):
        """set_offset sends VOLT:OFFS command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_offset(-0.5)
        awg.adapter.write.assert_called_with("VOLT:OFFS -0.5")

    def test_get_offset(self, awg):
        """get_offset queries offset."""
        awg.adapter.ask.return_value = "-0.5"
        result = awg.get_offset()
        awg.adapter.ask.assert_called_with("VOLT:OFFS?")
        assert result == -0.5

    def test_set_function_shape(self, awg):
        """set_function_shape sends FUNC:SHAP command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_function_shape(HP33120A.SHAPE_SINE)
        awg.adapter.write.assert_called_with("FUNC:SHAP SIN")

    def test_get_function_shape(self, awg):
        """get_function_shape queries shape."""
        awg.adapter.ask.return_value = "SIN"
        result = awg.get_function_shape()
        awg.adapter.ask.assert_called_with("FUNC:SHAP?")
        assert result == "SIN"

    def test_set_load_impedance_50(self, awg):
        """set_load_impedance(50) sends OUTP:LOAD 50."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_load_impedance(50)
        awg.adapter.write.assert_called_with("OUTP:LOAD 50")

    def test_set_load_impedance_high_z(self, awg):
        """set_load_impedance('INF') sends OUTP:LOAD INF."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_load_impedance(HP33120A.LOAD_HIGH_Z)
        awg.adapter.write.assert_called_with("OUTP:LOAD INF")

    def test_set_voltage_unit(self, awg):
        """set_voltage_unit sends VOLT:UNIT command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_voltage_unit(HP33120A.UNIT_VRMS)
        awg.adapter.write.assert_called_with("VOLT:UNIT VRMS")

    def test_get_voltage_unit(self, awg):
        """get_voltage_unit queries unit."""
        awg.adapter.ask.return_value = "VPP"
        result = awg.get_voltage_unit()
        assert result == "VPP"

    def test_set_duty_cycle(self, awg):
        """set_duty_cycle sends PULS:DCYC command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_duty_cycle(45)
        awg.adapter.write.assert_called_with("PULS:DCYC 45")

    def test_get_duty_cycle(self, awg):
        """get_duty_cycle queries duty cycle."""
        awg.adapter.ask.return_value = "45.0"
        result = awg.get_duty_cycle()
        assert result == 45.0

    def test_set_sync_output(self, awg):
        """set_sync_output sends OUTP:SYNC command."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sync_output(True)
        awg.adapter.write.assert_called_with("OUTP:SYNC ON")

    def test_get_sync_output(self, awg):
        """get_sync_output queries sync state."""
        awg.adapter.ask.return_value = "1"
        result = awg.get_sync_output()
        assert result is True


class TestHP33120AApplyMethods:
    """Test APPLy methods for standard waveforms."""

    def test_apply_sine_freq_only(self, awg):
        """apply_sine with frequency only."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_sine(1000)
        awg.adapter.write.assert_any_call("APPL:SIN 1000")

    def test_apply_sine_with_amplitude(self, awg):
        """apply_sine with frequency and amplitude."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_sine(1000, amplitude=2.0)
        awg.adapter.write.assert_any_call("APPL:SIN 1000, 2.0")

    def test_apply_sine_with_offset(self, awg):
        """apply_sine with all parameters."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_sine(1000, amplitude=2.0, offset=-0.5)
        awg.adapter.write.assert_any_call("APPL:SIN 1000, 2.0, -0.5")

    def test_apply_offset_requires_amplitude(self, awg):
        """apply_* raises error if offset specified without amplitude."""
        with pytest.raises(ValueError, match="amplitude must also be specified"):
            awg.apply_sine(1000, offset=-0.5)

    def test_apply_square(self, awg):
        """apply_square sends APPL:SQU."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_square(5000, amplitude=3.0)
        awg.adapter.write.assert_any_call("APPL:SQU 5000, 3.0")

    def test_apply_triangle(self, awg):
        """apply_triangle sends APPL:TRI."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_triangle(2000)
        awg.adapter.write.assert_any_call("APPL:TRI 2000")

    def test_apply_ramp(self, awg):
        """apply_ramp sends APPL:RAMP."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_ramp(500)
        awg.adapter.write.assert_any_call("APPL:RAMP 500")

    def test_apply_noise(self, awg):
        """apply_noise sends APPL:NOIS with DEF frequency."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_noise(amplitude=1.0)
        awg.adapter.write.assert_any_call("APPL:NOIS DEF, 1.0")

    def test_apply_dc(self, awg):
        """apply_dc sends APPL:DC with offset."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_dc(2.5)
        awg.adapter.write.assert_any_call("APPL:DC DEF, DEF, 2.5")

    def test_apply_user(self, awg):
        """apply_user sends APPL:USER."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.apply_user(1000, amplitude=1.5)
        awg.adapter.write.assert_any_call("APPL:USER 1000, 1.5")

    def test_get_apply_config(self, awg):
        """get_apply_config queries APPL?."""
        awg.adapter.ask.return_value = '"SIN 1000,2.0,0.0"'
        result = awg.get_apply_config()
        awg.adapter.ask.assert_called_with("APPL?")


class TestHP33120AWaveformUpload:
    """Test arbitrary waveform upload methods."""

    def test_upload_waveform_dac_numpy_array(self, awg):
        """upload_waveform_dac accepts numpy array."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        # Signed values from -2047 to +2047
        waveform = np.array([-2047, -1024, 0, 1024, 2047, 1024, 0, -1024], dtype=np.int16)
        awg.upload_waveform_dac(waveform)
        awg.adapter.write_binary.assert_called_once()
        args = awg.adapter.write_binary.call_args[0]
        assert args[0] == "DATA:DAC VOLATILE, "

    def test_upload_waveform_dac_with_name(self, awg):
        """upload_waveform_dac copies to non-volatile when name provided."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        waveform = np.linspace(-2047, 2047, 10, dtype=np.int16)
        awg.upload_waveform_dac(waveform, name="TEST")
        # Should have DATA:COPY call
        copy_calls = [c for c in awg.adapter.write.call_args_list if "DATA:COPY" in str(c)]
        assert len(copy_calls) == 1
        assert "TEST" in str(copy_calls[0])

    def test_upload_waveform_dac_list(self, awg):
        """upload_waveform_dac accepts list."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        waveform = [-2000, -1000, 0, 1000, 2000, 1000, 0, -1000]
        awg.upload_waveform_dac(waveform)
        awg.adapter.write_binary.assert_called_once()

    def test_upload_waveform_dac_invalid_range_positive(self, awg):
        """upload_waveform_dac rejects values > 2047."""
        waveform = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], dtype=np.int16)
        with pytest.raises(ValueError, match="-2047 to 2047"):
            awg.upload_waveform_dac(waveform)

    def test_upload_waveform_dac_invalid_range_negative(self, awg):
        """upload_waveform_dac rejects values < -2047."""
        waveform = np.array([-3000, -2048, -1000, 0, 1000, 2000, 1000, 0], dtype=np.int16)
        with pytest.raises(ValueError, match="-2047 to 2047"):
            awg.upload_waveform_dac(waveform)

    def test_upload_waveform_dac_too_few_points(self, awg):
        """upload_waveform_dac rejects < 8 points."""
        waveform = np.array([0, 100, 200], dtype=np.int16)
        with pytest.raises(ValueError, match="8-16000 points"):
            awg.upload_waveform_dac(waveform)

    def test_upload_waveform_dac_too_many_points(self, awg):
        """upload_waveform_dac rejects > 16000 points."""
        waveform = np.zeros(20000, dtype=np.int16)
        with pytest.raises(ValueError, match="8-16000 points"):
            awg.upload_waveform_dac(waveform)

    def test_upload_waveform_float(self, awg):
        """upload_waveform_float sends float format."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        waveform = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5])
        awg.upload_waveform_float(waveform)
        # Should have DATA VOLATILE call (not DATA:DAC)
        write_calls = [c for c in awg.adapter.write.call_args_list if "DATA VOLATILE" in str(c)]
        assert len(write_calls) == 1

    def test_upload_waveform_float_invalid_range(self, awg):
        """upload_waveform_float rejects values outside -1 to +1."""
        waveform = np.array([-1.5, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5])
        with pytest.raises(ValueError, match="-1.0 to \\+1.0"):
            awg.upload_waveform_float(waveform)

    def test_select_user_waveform(self, awg):
        """select_user_waveform sends FUNC:USER."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.select_user_waveform("MYWAVE")
        awg.adapter.write.assert_any_call("FUNC:USER MYWAVE")

    def test_list_waveforms(self, awg):
        """list_waveforms queries DATA:CAT?."""
        awg.adapter.ask.return_value = '"SINC,RAMP,NEG_RAMP,EXP_RISE,EXP_FALL"'
        result = awg.list_waveforms()
        awg.adapter.ask.assert_called_with("DATA:CAT?")

    def test_list_user_waveforms(self, awg):
        """list_user_waveforms queries DATA:NVOL:CAT?."""
        awg.adapter.ask.return_value = '"MYWAVE,TEST"'
        result = awg.list_user_waveforms()
        awg.adapter.ask.assert_called_with("DATA:NVOL:CAT?")

    def test_get_free_memory(self, awg):
        """get_free_memory queries DATA:NVOL:FREE?."""
        awg.adapter.ask.return_value = "32768"
        result = awg.get_free_memory()
        awg.adapter.ask.assert_called_with("DATA:NVOL:FREE?")
        assert result == 32768

    def test_get_waveform_points_volatile(self, awg):
        """get_waveform_points queries volatile by default."""
        awg.adapter.ask.return_value = "100"
        result = awg.get_waveform_points()
        awg.adapter.ask.assert_called_with("DATA:ATTR:POIN?")
        assert result == 100

    def test_get_waveform_points_named(self, awg):
        """get_waveform_points can query named waveform."""
        awg.adapter.ask.return_value = "500"
        result = awg.get_waveform_points("MYWAVE")
        awg.adapter.ask.assert_called_with("DATA:ATTR:POIN? MYWAVE")
        assert result == 500

    def test_delete_waveform(self, awg):
        """delete_waveform sends DATA:DEL."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.delete_waveform("MYWAVE")
        awg.adapter.write.assert_any_call("DATA:DEL MYWAVE")

    def test_delete_all_waveforms(self, awg):
        """delete_all_waveforms sends DATA:DEL:ALL."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.delete_all_waveforms()
        awg.adapter.write.assert_any_call("DATA:DEL:ALL")


class TestHP33120AModulationAM:
    """Test AM modulation methods."""

    def test_set_am_depth(self, awg):
        """set_am_depth sends AM:DEPT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_am_depth(50)
        awg.adapter.write.assert_called_with("AM:DEPT 50")

    def test_get_am_depth(self, awg):
        """get_am_depth queries AM:DEPT?."""
        awg.adapter.ask.return_value = "50.0"
        result = awg.get_am_depth()
        awg.adapter.ask.assert_called_with("AM:DEPT?")
        assert result == 50.0

    def test_set_am_source(self, awg):
        """set_am_source sends AM:SOUR."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_am_source(HP33120A.AM_SOURCE_BOTH)
        awg.adapter.write.assert_called_with("AM:SOUR BOTH")

    def test_get_am_source(self, awg):
        """get_am_source queries AM:SOUR?."""
        awg.adapter.ask.return_value = "BOTH"
        result = awg.get_am_source()
        assert result == "BOTH"

    def test_set_am_internal_function(self, awg):
        """set_am_internal_function sends AM:INT:FUNC."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_am_internal_function(HP33120A.SHAPE_SINE)
        awg.adapter.write.assert_called_with("AM:INT:FUNC SIN")

    def test_get_am_internal_function(self, awg):
        """get_am_internal_function queries AM:INT:FUNC?."""
        awg.adapter.ask.return_value = "SIN"
        result = awg.get_am_internal_function()
        assert result == "SIN"

    def test_set_am_internal_frequency(self, awg):
        """set_am_internal_frequency sends AM:INT:FREQ."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_am_internal_frequency(100)
        awg.adapter.write.assert_called_with("AM:INT:FREQ 100")

    def test_get_am_internal_frequency(self, awg):
        """get_am_internal_frequency queries AM:INT:FREQ?."""
        awg.adapter.ask.return_value = "100.0"
        result = awg.get_am_internal_frequency()
        assert result == 100.0

    def test_set_am_state(self, awg):
        """set_am_state sends AM:STAT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_am_state(True)
        awg.adapter.write.assert_called_with("AM:STAT ON")

    def test_get_am_state(self, awg):
        """get_am_state queries AM:STAT?."""
        awg.adapter.ask.return_value = "1"
        result = awg.get_am_state()
        assert result is True


class TestHP33120AModulationFM:
    """Test FM modulation methods."""

    def test_set_fm_deviation(self, awg):
        """set_fm_deviation sends FM:DEV."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fm_deviation(1000)
        awg.adapter.write.assert_called_with("FM:DEV 1000")

    def test_get_fm_deviation(self, awg):
        """get_fm_deviation queries FM:DEV?."""
        awg.adapter.ask.return_value = "1000.0"
        result = awg.get_fm_deviation()
        assert result == 1000.0

    def test_set_fm_internal_function(self, awg):
        """set_fm_internal_function sends FM:INT:FUNC."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fm_internal_function(HP33120A.SHAPE_TRIANGLE)
        awg.adapter.write.assert_called_with("FM:INT:FUNC TRI")

    def test_set_fm_internal_frequency(self, awg):
        """set_fm_internal_frequency sends FM:INT:FREQ."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fm_internal_frequency(10)
        awg.adapter.write.assert_called_with("FM:INT:FREQ 10")

    def test_set_fm_state(self, awg):
        """set_fm_state sends FM:STAT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fm_state(True)
        awg.adapter.write.assert_called_with("FM:STAT ON")


class TestHP33120AModulationBurst:
    """Test burst modulation methods."""

    def test_set_burst_cycles(self, awg):
        """set_burst_cycles sends BM:NCYC."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_cycles(5)
        awg.adapter.write.assert_called_with("BM:NCYC 5")

    def test_set_burst_cycles_infinite(self, awg):
        """set_burst_cycles accepts INF."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_cycles("INF")
        awg.adapter.write.assert_called_with("BM:NCYC INF")

    def test_get_burst_cycles(self, awg):
        """get_burst_cycles queries BM:NCYC?."""
        awg.adapter.ask.return_value = "5"
        result = awg.get_burst_cycles()
        assert result == "5"

    def test_set_burst_phase(self, awg):
        """set_burst_phase sends BM:PHAS."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_phase(90)
        awg.adapter.write.assert_called_with("BM:PHAS 90")

    def test_get_burst_phase(self, awg):
        """get_burst_phase queries BM:PHAS?."""
        awg.adapter.ask.return_value = "90.0"
        result = awg.get_burst_phase()
        assert result == 90.0

    def test_set_burst_internal_rate(self, awg):
        """set_burst_internal_rate sends BM:INT:RATE."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_internal_rate(100)
        awg.adapter.write.assert_called_with("BM:INT:RATE 100")

    def test_set_burst_source(self, awg):
        """set_burst_source sends BM:SOUR."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_source(HP33120A.SOURCE_INTERNAL)
        awg.adapter.write.assert_called_with("BM:SOUR INT")

    def test_set_burst_state(self, awg):
        """set_burst_state sends BM:STAT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_burst_state(True)
        awg.adapter.write.assert_called_with("BM:STAT ON")


class TestHP33120AFSK:
    """Test FSK methods."""

    def test_set_fsk_frequency(self, awg):
        """set_fsk_frequency sends FSK:FREQ."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fsk_frequency(2000)
        awg.adapter.write.assert_called_with("FSK:FREQ 2000")

    def test_get_fsk_frequency(self, awg):
        """get_fsk_frequency queries FSK:FREQ?."""
        awg.adapter.ask.return_value = "2000.0"
        result = awg.get_fsk_frequency()
        assert result == 2000.0

    def test_set_fsk_internal_rate(self, awg):
        """set_fsk_internal_rate sends FSK:INT:RATE."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fsk_internal_rate(10)
        awg.adapter.write.assert_called_with("FSK:INT:RATE 10")

    def test_set_fsk_source(self, awg):
        """set_fsk_source sends FSK:SOUR."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fsk_source(HP33120A.SOURCE_EXTERNAL)
        awg.adapter.write.assert_called_with("FSK:SOUR EXT")

    def test_set_fsk_state(self, awg):
        """set_fsk_state sends FSK:STAT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_fsk_state(True)
        awg.adapter.write.assert_called_with("FSK:STAT ON")


class TestHP33120ASweep:
    """Test sweep methods."""

    def test_set_sweep_start_frequency(self, awg):
        """set_sweep_start_frequency sends FREQ:STAR."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sweep_start_frequency(100)
        awg.adapter.write.assert_called_with("FREQ:STAR 100")

    def test_get_sweep_start_frequency(self, awg):
        """get_sweep_start_frequency queries FREQ:STAR?."""
        awg.adapter.ask.return_value = "100.0"
        result = awg.get_sweep_start_frequency()
        assert result == 100.0

    def test_set_sweep_stop_frequency(self, awg):
        """set_sweep_stop_frequency sends FREQ:STOP."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sweep_stop_frequency(10000)
        awg.adapter.write.assert_called_with("FREQ:STOP 10000")

    def test_set_sweep_spacing(self, awg):
        """set_sweep_spacing sends SWE:SPAC."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sweep_spacing(HP33120A.SWEEP_LOGARITHMIC)
        awg.adapter.write.assert_called_with("SWE:SPAC LOG")

    def test_get_sweep_spacing(self, awg):
        """get_sweep_spacing queries SWE:SPAC?."""
        awg.adapter.ask.return_value = "LOG"
        result = awg.get_sweep_spacing()
        assert result == "LOG"

    def test_set_sweep_time(self, awg):
        """set_sweep_time sends SWE:TIME."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sweep_time(5.0)
        awg.adapter.write.assert_called_with("SWE:TIME 5.0")

    def test_set_sweep_state(self, awg):
        """set_sweep_state sends SWE:STAT."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_sweep_state(True)
        awg.adapter.write.assert_called_with("SWE:STAT ON")


class TestHP33120ATrigger:
    """Test trigger methods."""

    def test_set_trigger_source(self, awg):
        """set_trigger_source sends TRIG:SOUR."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_trigger_source(HP33120A.TRIGGER_BUS)
        awg.adapter.write.assert_called_with("TRIG:SOUR BUS")

    def test_get_trigger_source(self, awg):
        """get_trigger_source queries TRIG:SOUR?."""
        awg.adapter.ask.return_value = "BUS"
        result = awg.get_trigger_source()
        assert result == "BUS"

    def test_set_trigger_slope(self, awg):
        """set_trigger_slope sends TRIG:SLOP."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_trigger_slope(HP33120A.SLOPE_NEGATIVE)
        awg.adapter.write.assert_called_with("TRIG:SLOP NEG")

    def test_get_trigger_slope(self, awg):
        """get_trigger_slope queries TRIG:SLOP?."""
        awg.adapter.ask.return_value = "NEG"
        result = awg.get_trigger_slope()
        assert result == "NEG"

    def test_trigger(self, awg):
        """trigger sends *TRG."""
        awg.trigger()
        awg.adapter.write.assert_called_with("*TRG")


class TestHP33120ALegacyMethods:
    """Test legacy/deprecated methods for backwards compatibility."""

    def test_upload_waveform_legacy(self, awg):
        """Legacy upload_waveform still works."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        waveform = np.linspace(-2047, 2047, 10, dtype=np.int16)
        awg.upload_waveform(waveform, name="TEST")
        awg.adapter.write_binary.assert_called_once()

    def test_select_waveform_legacy(self, awg):
        """Legacy select_waveform still works."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.select_waveform("TEST")
        awg.adapter.write.assert_any_call("FUNC:USER TEST")

    def test_set_function_shape_user_legacy(self, awg):
        """Legacy set_function_shape_user still works."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.set_function_shape_user()
        awg.adapter.write.assert_called_with("FUNC:SHAP USER")

    def test_configure_output_legacy(self, awg):
        """Legacy configure_output still works."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        awg.configure_output(frequency=1000, voltage=2.0, load=50)
        # Should have multiple writes
        write_calls = awg.adapter.write.call_args_list
        assert any("OUTP:LOAD" in str(c) for c in write_calls)
        assert any("FREQ" in str(c) for c in write_calls)
        assert any("VOLT" in str(c) for c in write_calls)

    def test_setup_arbitrary_waveform_legacy(self, awg):
        """Legacy setup_arbitrary_waveform still works."""
        awg.adapter.ask.return_value = "+0,\"No error\""
        waveform = np.linspace(-2047, 2047, 10, dtype=np.int16)
        awg.setup_arbitrary_waveform(waveform, name="TEST", frequency=1000, voltage=2.0)
        # Should have binary write and multiple commands
        awg.adapter.write_binary.assert_called_once()
