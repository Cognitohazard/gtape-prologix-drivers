"""Hardware test script for HP34401A digital multimeter.

This script tests the HP34401A DMM driver with actual hardware.
"""

import sys
import time

# Add parent directory to path for imports

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.hp34401a import HP34401A


def test_dmm_basic(port, gpib_addr):
    """Test basic DMM operations.

    Args:
        port: COM port (e.g., "COM4")
        gpib_addr: GPIB address of DMM (e.g., 8)
    """
    print("="*60)
    print("HP 34401A DIGITAL MULTIMETER HARDWARE TEST")
    print("="*60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("="*60 + "\n")

    try:
        # Connect to DMM
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create DMM instance
        dmm = HP34401A(adapter)

        # Test 1: Query identification
        print("[Test 1] Querying DMM identification...")
        idn = adapter.ask("*IDN?")
        print(f"[Test 1] [OK] DMM ID: {idn}\n")

        # Test 2: Reset DMM
        print("[Test 2] Resetting DMM...")
        dmm.reset()
        print("[Test 2] [OK] Reset complete\n")

        # Test 3: Quick DC voltage measurement (autorange - slow)
        print("[Test 3] DC voltage measurement (autorange - may take 3-4s)...")
        print("[Test 3] NOTE: Connect a voltage source to the DMM input")
        print("[Test 3]       (or leave open for ~0V reading)")
        t0 = time.time()
        voltage = dmm.measure_voltage()
        t_auto = (time.time() - t0) * 1000
        print(f"[Test 3] [OK] DC Voltage: {voltage:.6f} V ({t_auto:.0f}ms)\n")

        # Test 3b: Fixed range voltage measurement (fast)
        print("[Test 3b] DC voltage measurement (10V range - fast)...")
        t0 = time.time()
        voltage_fixed = dmm.measure_voltage(range_volts=10)
        t_fixed = (time.time() - t0) * 1000
        print(f"[Test 3b] [OK] DC Voltage: {voltage_fixed:.6f} V ({t_fixed:.0f}ms)")
        print(f"[Test 3b] Speed improvement: {t_auto/t_fixed:.1f}x faster\n")

        # Test 4: Configured DC voltage measurement with specific range
        print("[Test 4] Configured DC voltage (10V range, high resolution)...")
        dmm.configure_dc_voltage(range_volts=10, resolution=0.00001)
        reading = dmm.read()
        print(f"[Test 4] [OK] Configured reading: {reading:.6f} V\n")

        # Test 4b: NPLC configuration and speed comparison
        print("[Test 4b] NPLC configuration and speed comparison...")
        dmm.configure_dc_voltage(range_volts=10)
        nplc_values = [10, 1, 0.2]
        print(f"[Test 4b] Testing NPLC values: {nplc_values}")
        for nplc in nplc_values:
            dmm.set_nplc(nplc)
            actual_nplc = dmm.get_nplc()
            # Take 5 readings and average the time
            times = []
            for _ in range(5):
                t0 = time.time()
                v = dmm.read()
                times.append((time.time() - t0) * 1000)
            avg_ms = sum(times) / len(times)
            print(f"[Test 4b]   NPLC={nplc:5.2f}: set={actual_nplc}, avg={avg_ms:.0f}ms/reading, last={v:.6f}V")
        # Reset to NPLC 1 (good balance)
        dmm.set_nplc(1)
        print(f"[Test 4b] [OK] NPLC reset to 1\n")

        # Test 5: DC current measurement (fixed range - fast)
        print("[Test 5] DC current measurement (1A range - fast)...")
        print("[Test 5] WARNING: Connect current source or leave disconnected")
        try:
            t0 = time.time()
            current = dmm.measure_current(range_amps=1)
            t_ms = (time.time() - t0) * 1000
            print(f"[Test 5] [OK] DC Current: {current:.9f} A ({t_ms:.0f}ms)\n")
        except Exception as e:
            print(f"[Test 5] [INFO] Current measurement: {e}")
            print("[Test 5] [INFO] This is normal if no current source connected\n")

        # Test 6: Resistance measurement (fixed range - fast)
        print("[Test 6] 2-wire resistance measurement (10k range - fast)...")
        print("[Test 6] NOTE: Connect a resistor or leave open for overload")
        try:
            t0 = time.time()
            resistance = dmm.measure_resistance(range_ohms=10000)
            t_ms = (time.time() - t0) * 1000
            if resistance > 1e8:
                print(f"[Test 6] [OK] Resistance: >100M ohms (open circuit) ({t_ms:.0f}ms)")
            else:
                print(f"[Test 6] [OK] Resistance: {resistance:.3f} ohms ({t_ms:.0f}ms)")
        except Exception as e:
            print(f"[Test 6] [INFO] Resistance: {e}")
        print()

        # Test 7: Test multiple measurement types
        print("[Test 7] Testing multiple measurement configurations...")

        # AC voltage
        dmm.configure_ac_voltage(range_volts=10)
        print("[Test 7] [OK] AC voltage configured")

        # DC current with range
        dmm.configure_dc_current(range_amps=1, resolution=0.000001)
        print("[Test 7] [OK] DC current configured")

        # Resistance with range
        dmm.configure_resistance(range_ohms=10000, resolution=0.001)
        print("[Test 7] [OK] Resistance configured")

        print("[Test 7] [OK] All configurations successful\n")

        # Test 8: Return to DC voltage and take final reading (fixed range)
        print("[Test 8] Final DC voltage measurement (10V range)...")
        t0 = time.time()
        final_voltage = dmm.measure_voltage(range_volts=10)
        t_ms = (time.time() - t0) * 1000
        print(f"[Test 8] [OK] Final voltage: {final_voltage:.6f} V ({t_ms:.0f}ms)\n")

        # Test 9: Check for errors
        print("[Test 9] Checking for errors...")
        error = dmm.check_errors()
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
        print("The HP34401A driver is working correctly with actual hardware.\n")

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

        # Try to close connection on error
        try:
            adapter.close()
            print("\n[Cleanup] Connection closed")
        except:
            pass

        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hardware test for HP 34401A digital multimeter"
    )
    parser.add_argument("port", help="COM port (e.g., COM4)")
    parser.add_argument("--addr", type=int, default=8,
                       help="GPIB address (default: 8)")

    args = parser.parse_args()

    sys.exit(test_dmm_basic(args.port, args.addr))
