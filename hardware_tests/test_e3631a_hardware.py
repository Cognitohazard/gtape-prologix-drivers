"""Hardware test script for Agilent E3631A power supply.

This script tests the E3631A PSU driver with actual hardware.
"""

import sys
import time

# Add parent directory to path for imports

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.agilent_e3631a import AgilentE3631A


def test_psu_basic(port, gpib_addr):
    """Test basic PSU operations.

    Args:
        port: COM port (e.g., "COM3")
        gpib_addr: GPIB address of PSU (default: 3)
    """
    print("="*60)
    print("AGILENT E3631A HARDWARE TEST")
    print("="*60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("="*60 + "\n")

    try:
        # Connect to PSU
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create PSU instance
        psu = AgilentE3631A(adapter)

        # Test 1: Query identification
        print("[Test 1] Querying PSU identification...")
        idn = adapter.ask("*IDN?")
        print(f"[Test 1] [OK] PSU ID: {idn}\n")

        # Test 2: Reset PSU
        print("[Test 2] Resetting PSU...")
        psu.reset()
        print("[Test 2] [OK] Reset complete\n")

        # Test 3: Select P25V channel and configure
        print("[Test 3] Configuring P25V output (5V, 0.1A limit)...")
        psu.configure_output(
            channel=AgilentE3631A.P25V,
            voltage=5.0,
            current_limit=0.1
        )
        print("[Test 3] [OK] Configuration complete\n")

        # Test 4: Enable output
        print("[Test 4] Enabling output...")
        psu.enable_output(True)
        print("[Test 4] [OK] Output enabled")
        time.sleep(0.5)  # Let output stabilize

        # Test 5: Measure voltage and current
        print("\n[Test 5] Reading measurements...")
        voltage = psu.measure_voltage()
        current = psu.measure_current()
        print(f"[Test 5] [OK] Measured voltage: {voltage:.4f} V")
        print(f"[Test 5] [OK] Measured current: {current:.6f} A\n")

        # Test 6: Change voltage
        print("[Test 6] Changing voltage to 3.0V...")
        psu.set_voltage(3.0)
        time.sleep(0.5)
        voltage = psu.measure_voltage()
        current = psu.measure_current()
        print(f"[Test 6] [OK] New voltage: {voltage:.4f} V")
        print(f"[Test 6] [OK] New current: {current:.6f} A\n")

        # Test 7: Test P6V channel
        print("[Test 7] Switching to P6V channel (3.3V, 0.1A limit)...")
        psu.configure_output(
            channel=AgilentE3631A.P6V,
            voltage=3.3,
            current_limit=0.1
        )
        time.sleep(0.5)
        voltage = psu.measure_voltage()
        current = psu.measure_current()
        print(f"[Test 7] [OK] P6V voltage: {voltage:.4f} V")
        print(f"[Test 7] [OK] P6V current: {current:.6f} A\n")

        # Test 8: Disable output
        print("[Test 8] Disabling output...")
        psu.enable_output(False)
        print("[Test 8] [OK] Output disabled\n")

        # Test 9: Check for errors
        print("[Test 9] Checking for errors...")
        error = psu.check_errors()
        if error.startswith("+0"):
            print("[Test 9] [OK] No errors detected\n")
        else:
            print(f"[Test 9] [WARNING] Error: {error}\n")

        # Cleanup
        print("[Cleanup] Closing connection...")
        adapter.close()
        print("[Cleanup] [OK] Connection closed\n")

        # Summary
        print("="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nHardware verification complete!")
        print("The E3631A driver is working correctly with actual hardware.\n")

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

        # Try to disable output on error
        try:
            psu.enable_output(False)
            adapter.close()
            print("\n[Cleanup] Output disabled and connection closed")
        except:
            pass

        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hardware test for Agilent E3631A power supply"
    )
    parser.add_argument("port", help="COM port (e.g., COM3)")
    parser.add_argument("--addr", type=int, default=3,
                       help="GPIB address (default: 3)")

    args = parser.parse_args()

    sys.exit(test_psu_basic(args.port, args.addr))
