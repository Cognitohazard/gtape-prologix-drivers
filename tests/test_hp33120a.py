"""Unit tests for HP33120A AWG driver."""

import unittest
from unittest.mock import Mock, MagicMock, call, patch
import numpy as np
from gtape_prologix_drivers.instruments.hp33120a import HP33120A


class TestHP33120A(unittest.TestCase):
    """Test cases for HP33120A AWG driver."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.awg = HP33120A(self.mock_adapter)

    def test_initialization(self):
        """Test AWG initialization."""
        self.assertEqual(self.awg.adapter, self.mock_adapter)

    def test_reset(self):
        """Test AWG reset."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        self.awg.reset()

        # Verify *RST was sent
        self.mock_adapter.write.assert_called_with("*RST")

        # Verify error check was performed
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_upload_waveform_numpy_array(self):
        """Test uploading waveform from numpy array."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        # Create simple waveform (minimum 8 points required)
        waveform = np.array([0, 292, 585, 877, 1170, 1462, 1755, 2047], dtype=np.uint16)

        self.awg.upload_waveform(waveform, name="TEST")

        # Verify DATA:DAC VOLATILE was called with binary data
        calls = self.mock_adapter.write_binary.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "DATA:DAC VOLATILE, ")

        # Verify DATA:COPY was called
        copy_call = [c for c in self.mock_adapter.write.call_args_list
                     if "DATA:COPY" in str(c)]
        self.assertEqual(len(copy_call), 1)
        self.assertIn("TEST", str(copy_call[0]))

    def test_upload_waveform_list(self):
        """Test uploading waveform from list."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        # Create waveform as list (minimum 8 points required)
        waveform = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]

        self.awg.upload_waveform(waveform, name="MYWAVE")

        # Verify upload succeeded
        self.mock_adapter.write_binary.assert_called_once()

    def test_upload_waveform_invalid_range(self):
        """Test error when waveform values out of range."""
        # Values above DAC_MAX (must have >= 8 points)
        waveform = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], dtype=np.uint16)

        with self.assertRaises(ValueError) as cm:
            self.awg.upload_waveform(waveform)

        self.assertIn("0-2047", str(cm.exception))

    def test_upload_waveform_too_few_points(self):
        """Test error when too few points."""
        waveform = np.array([0, 100], dtype=np.uint16)  # Only 2 points < 8

        with self.assertRaises(ValueError) as cm:
            self.awg.upload_waveform(waveform)

        self.assertIn("8-16000 points", str(cm.exception))

    def test_upload_waveform_too_many_points(self):
        """Test error when too many points."""
        waveform = np.zeros(20000, dtype=np.uint16)  # 20000 > 16000

        with self.assertRaises(ValueError) as cm:
            self.awg.upload_waveform(waveform)

        self.assertIn("8-16000 points", str(cm.exception))

    def test_select_waveform(self):
        """Test selecting waveform from memory."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        self.awg.select_waveform("MYWAVE")

        # Verify FUNC:USER command was sent
        self.mock_adapter.write.assert_called_with("FUNC:USER MYWAVE")
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_set_function_shape_user(self):
        """Test setting function shape to USER."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        self.awg.set_function_shape_user()

        self.mock_adapter.write.assert_called_with("FUNC:SHAP USER")
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_configure_output_default(self):
        """Test output configuration with default parameters."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        self.awg.configure_output()

        # Check calls were made
        calls = self.mock_adapter.write.call_args_list

        # Should have OUTP:LOAD and FREQ;VOLT commands
        outp_calls = [c for c in calls if "OUTP:LOAD" in str(c)]
        freq_calls = [c for c in calls if "FREQ" in str(c)]

        self.assertEqual(len(outp_calls), 1)
        self.assertEqual(len(freq_calls), 1)

        # Verify default values
        self.assertIn("50", str(outp_calls[0]))  # Default load
        self.assertIn("5000", str(freq_calls[0]))  # Default frequency
        self.assertIn("0.5", str(freq_calls[0]))   # Default voltage

    def test_configure_output_custom(self):
        """Test output configuration with custom parameters."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        self.awg.configure_output(frequency=10000, voltage=1.0, load=600)

        calls = self.mock_adapter.write.call_args_list

        # Find FREQ command
        freq_calls = [c for c in calls if "FREQ" in str(c)]
        self.assertEqual(len(freq_calls), 1)
        self.assertIn("10000", str(freq_calls[0]))
        self.assertIn("1.0", str(freq_calls[0]))

        # Find OUTP command
        outp_calls = [c for c in calls if "OUTP:LOAD" in str(c)]
        self.assertIn("600", str(outp_calls[0]))

    def test_check_errors_no_error(self):
        """Test error checking with no errors."""
        self.mock_adapter.ask.return_value = '+0,"No error"'

        result = self.awg.check_errors()

        self.assertEqual(result, '+0,"No error"')
        self.mock_adapter.ask.assert_called_with("SYST:ERR?")

    def test_check_errors_with_error(self):
        """Test error checking when error exists."""
        self.mock_adapter.ask.return_value = '-113,"Undefined header"'

        result = self.awg.check_errors()

        self.assertEqual(result, '-113,"Undefined header"')

    def test_setup_arbitrary_waveform(self):
        """Test complete waveform setup workflow."""
        self.mock_adapter.ask.return_value = "+0,\"No error\""

        waveform = np.linspace(0, 2047, 100, dtype=np.uint16)

        self.awg.setup_arbitrary_waveform(
            waveform,
            name="RAMP",
            frequency=8000,
            voltage=0.8,
            load=50
        )

        # Verify all steps were performed
        # 1. Upload (write_binary)
        self.mock_adapter.write_binary.assert_called_once()

        # 2. Select (FUNC:USER)
        func_user_calls = [c for c in self.mock_adapter.write.call_args_list
                           if "FUNC:USER" in str(c)]
        self.assertEqual(len(func_user_calls), 1)
        self.assertIn("RAMP", str(func_user_calls[0]))

        # 3. Set shape (FUNC:SHAP USER)
        func_shap_calls = [c for c in self.mock_adapter.write.call_args_list
                           if "FUNC:SHAP USER" in str(c)]
        self.assertEqual(len(func_shap_calls), 1)

        # 4. Configure output
        freq_calls = [c for c in self.mock_adapter.write.call_args_list
                      if "FREQ" in str(c)]
        self.assertEqual(len(freq_calls), 1)
        self.assertIn("8000", str(freq_calls[0]))
        self.assertIn("0.8", str(freq_calls[0]))

    def test_dac_constants(self):
        """Test DAC value constants."""
        self.assertEqual(HP33120A.DAC_MIN, 0)
        self.assertEqual(HP33120A.DAC_MAX, 2047)
        self.assertEqual(HP33120A.MIN_POINTS, 8)
        self.assertEqual(HP33120A.MAX_POINTS, 16000)


if __name__ == '__main__':
    unittest.main()
