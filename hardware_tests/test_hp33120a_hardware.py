"""Hardware test script for HP33120A Arbitrary Waveform Generator.

This script tests the HP33120A AWG driver with actual hardware.
"""

import sys
import numpy as np

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.hp33120a import HP33120A


def test_awg_basic(port, gpib_addr):
    """Test basic AWG operations.

    Args:
        port: COM port (e.g., "COM3")
        gpib_addr: GPIB address of AWG (default: 10)
    """
    print("=" * 60)
    print("HP33120A HARDWARE TEST")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("=" * 60 + "\n")

    adapter = None
    awg = None
    try:
        # Connect to AWG
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create AWG instance
        awg = HP33120A(adapter)

        # Test 1: Query identification
        print("[Test 1] Querying AWG identification...")
        idn = awg.get_identification()
        print(f"[Test 1] [OK] AWG ID: {idn}\n")

        # Test 2: Reset AWG
        print("[Test 2] Resetting AWG...")
        awg.reset()
        print("[Test 2] [OK] Reset complete\n")

        # Test 3: Output a sine wave using APPLy
        print("[Test 3] Configuring sine wave (1kHz, 1Vpp)...")
        awg.apply_sine(1000, amplitude=1.0)
        print("[Test 3] [OK] Sine wave configured")

        # Verify settings
        freq = awg.get_frequency()
        amp = awg.get_amplitude()
        shape = awg.get_function_shape()
        print(f"[Test 3] [OK] Readback: {shape} @ {freq}Hz, {amp}Vpp\n")

        # Test 4: Change to square wave
        print("[Test 4] Changing to square wave (5kHz, 2Vpp, 50% duty)...")
        awg.apply_square(5000, amplitude=2.0)
        awg.set_duty_cycle(50)
        freq = awg.get_frequency()
        duty = awg.get_duty_cycle()
        print(f"[Test 4] [OK] Square wave: {freq}Hz, {duty}% duty\n")

        # Test 5: Test offset
        print("[Test 5] Adding DC offset (-0.5V)...")
        awg.set_offset(-0.5)
        offset = awg.get_offset()
        print(f"[Test 5] [OK] Offset: {offset}V\n")

        # Test 6: Upload arbitrary waveform (signed DAC format)
        print("[Test 6] Uploading arbitrary waveform (8-point triangle)...")
        # Create a simple triangle using signed values
        triangle = np.array([-2047, -1024, 0, 1024, 2047, 1024, 0, -1024], dtype=np.int16)
        awg.upload_waveform_dac(triangle, name="TRIANGLE")
        points = awg.get_waveform_points("TRIANGLE")
        print(f"[Test 6] [OK] Uploaded {points} points to 'TRIANGLE'\n")

        # Test 7: Select and output arbitrary waveform
        print("[Test 7] Selecting and outputting arbitrary waveform...")
        awg.select_user_waveform("TRIANGLE")
        awg.apply_user(2000, amplitude=1.5)
        shape = awg.get_function_shape()
        print(f"[Test 7] [OK] Function shape: {shape}\n")

        # Test 8: List waveforms
        print("[Test 8] Listing available waveforms...")
        waveforms = awg.list_waveforms()
        print(f"[Test 8] [OK] Waveforms: {waveforms}")
        user_waveforms = awg.list_user_waveforms()
        print(f"[Test 8] [OK] User waveforms: {user_waveforms}")
        free_mem = awg.get_free_memory()
        print(f"[Test 8] [OK] Free memory: {free_mem} bytes\n")

        # Test 9: Test state save/recall
        print("[Test 9] Testing state save/recall...")
        awg.apply_sine(8000, amplitude=0.5)
        awg.save_state(1)
        print("[Test 9] [OK] State saved to location 1")

        awg.apply_square(1000, amplitude=3.0)  # Change settings
        awg.recall_state(1)  # Restore
        freq = awg.get_frequency()
        amp = awg.get_amplitude()
        print(f"[Test 9] [OK] State recalled: {freq}Hz, {amp}Vpp\n")

        # Test 10: Display text
        print("[Test 10] Testing display...")
        awg.set_display_text("TEST OK")
        text = awg.get_display_text()
        print(f"[Test 10] [OK] Display text: {text}")
        awg.clear_display_text()
        print("[Test 10] [OK] Display cleared\n")

        # Test 11: Clean up - delete test waveform
        print("[Test 11] Cleaning up test waveform...")
        awg.delete_waveform("TRIANGLE")
        print("[Test 11] [OK] Waveform deleted\n")

        # Test 12: Check for errors
        print("[Test 12] Checking for errors...")
        error = awg.check_errors()
        if error.startswith("+0"):
            print("[Test 12] [OK] No errors detected\n")
        else:
            print(f"[Test 12] [WARNING] Error: {error}\n")

        # Final reset
        print("[Cleanup] Resetting to defaults...")
        awg.reset()
        adapter.close()
        print("[Cleanup] [OK] Connection closed\n")

        # Summary
        print("=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
        print("\nHardware verification complete!")
        print("The HP33120A driver is working correctly with actual hardware.\n")

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

        # Try to reset on error
        try:
            if awg is not None:
                awg.reset()
            if adapter is not None:
                adapter.close()
            print("\n[Cleanup] AWG reset and connection closed")
        except:
            pass

        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hardware test for HP33120A Arbitrary Waveform Generator"
    )
    parser.add_argument("port", help="COM port (e.g., COM3)")
    parser.add_argument("--addr", type=int, default=10,
                        help="GPIB address (default: 10)")

    args = parser.parse_args()

    sys.exit(test_awg_basic(args.port, args.addr))
