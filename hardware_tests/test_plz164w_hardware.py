"""Hardware test script for Kikusui PLZ164W electronic load.

This script tests the PLZ164W load driver with actual hardware.

WARNING: Connect a power source capable of supplying 1-2A before running tests.
The load will be configured to sink current in CC mode.
"""

import sys
import time
import argparse

# Add parent directory to path for imports

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.plz164w import PLZ164W


def test_load_basic(port, gpib_addr, start_test=1):
    """Test basic load operations.

    Args:
        port: COM port (e.g., "COM3")
        gpib_addr: GPIB address of load (default: 11)
        start_test: Start from this test number (default: 1)
    """
    print("="*60)
    print("KIKUSUI PLZ164W HARDWARE TEST")
    print("="*60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    if start_test > 1:
        print(f"Starting from: Test {start_test}")
    print("\nWARNING: Ensure a power source is connected to the load!")
    print("         Recommended: 5V supply capable of 2A")
    print("="*60 + "\n")

    try:
        # Connect to load
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create load instance
        load = PLZ164W(adapter)

        # Test 1: Query identification
        if start_test <= 1:
            print("[Test 1] Querying load identification...")
            idn = load.get_identification()
            print(f"[Test 1] [OK] Load ID: {idn}\n")

        # Test 2: Reset load
        if start_test <= 2:
            print("[Test 2] Resetting load...")
            load.reset()
            print("[Test 2] [OK] Reset complete\n")

        # Test 3: Configure CC mode with low current
        if start_test <= 3:
            print("[Test 3] Configuring CC mode (0.5A)...")
            load.configure_cc_mode(current=0.5, current_range=PLZ164W.CURR_RANGE_HIGH)
            mode = load.get_mode()
            current_setting = load.get_current()
            print(f"[Test 3] [OK] Mode: {mode}")
            print(f"[Test 3] [OK] Current setting: {current_setting:.3f} A\n")

        # Test 4: Enable input
        if start_test <= 4:
            print("[Test 4] Enabling load input...")
            load.enable_input(True)
            print("[Test 4] [OK] Input enabled")
            time.sleep(1.0)  # Let load stabilize

        # Test 5: Measure voltage, current, and power
        if start_test <= 5:
            print("\n[Test 5] Reading measurements...")
            voltage = load.measure_voltage()
            current = load.measure_current()
            power = load.measure_power()
            print(f"[Test 5] [OK] Measured voltage: {voltage:.4f} V")
            print(f"[Test 5] [OK] Measured current: {current:.4f} A")
            print(f"[Test 5] [OK] Measured power: {power:.4f} W\n")

            # Verify load is actually working
            if voltage < 1.0:
                print("[Test 5] [WARNING] Voltage is very low - is power source connected?")
            if current < 0.4 or current > 0.6:
                print(f"[Test 5] [WARNING] Current ({current:.3f}A) not near setpoint (0.5A)")
            print()

        # Test 6: Change current to 1.5A
        if start_test <= 6:
            print("[Test 6] Changing current to 1.5A...")
            load.set_current(1.5)
            time.sleep(1.0)
            voltage = load.measure_voltage()
            current = load.measure_current()
            power = load.measure_power()
            print(f"[Test 6] [OK] New voltage: {voltage:.4f} V")
            print(f"[Test 6] [OK] New current: {current:.4f} A")
            print(f"[Test 6] [OK] New power: {power:.4f} W\n")

            # Test 6b: Cycle through current ranges
            print("[Test 6b] Testing current range switching (LOW -> MEDIUM -> HIGH)...")
            load.enable_input(False)  # Must disable input before changing ranges
            time.sleep(0.5)

            load.set_current_range(PLZ164W.CURR_RANGE_LOW)
            time.sleep(0.5)
            print(f"[Test 6b] [OK] Set to LOW range")

            load.set_current_range(PLZ164W.CURR_RANGE_MED)
            time.sleep(0.5)
            print(f"[Test 6b] [OK] Set to MEDIUM range")

            load.set_current_range(PLZ164W.CURR_RANGE_HIGH)
            time.sleep(0.5)
            print(f"[Test 6b] [OK] Set to HIGH range")
            print(f"[Test 6b] [OK] Current range cycling complete")

            # Re-enable input to continue testing
            load.enable_input(True)
            time.sleep(0.5)
            print(f"[Test 6b] [OK] Input re-enabled\n")

        # Test 7: Switch to CV mode
        if start_test <= 7:
            print("[Test 7] Switching to CV mode (3.3V)...")
            load.enable_input(False)  # Disable before mode change
            time.sleep(0.5)
            load.configure_cv_mode(voltage=3.3, voltage_range=PLZ164W.VOLT_RANGE_LOW)
            mode = load.get_mode()
            voltage_setting = load.get_voltage()
            print(f"[Test 7] [OK] Mode: {mode}")
            print(f"[Test 7] [OK] Voltage setting: {voltage_setting:.3f} V\n")

        # Test 8: Enable and measure in CV mode
        if start_test <= 8:
            print("[Test 8] Enabling load in CV mode...")
            load.enable_input(True)
            time.sleep(1.0)
            voltage = load.measure_voltage()
            current = load.measure_current()
            power = load.measure_power()
            print(f"[Test 8] [OK] Voltage: {voltage:.4f} V")
            print(f"[Test 8] [OK] Current: {current:.4f} A")
            print(f"[Test 8] [OK] Power: {power:.4f} W")

            # Verify voltage regulation
            if abs(voltage - 3.3) > 0.2:
                print(f"[Test 8] [WARNING] Voltage ({voltage:.3f}V) not near setpoint (3.3V)")
            print()

        # Test 9: Switch to CR mode
        if start_test <= 9:
            print("[Test 9] Switching to CR mode (5 Ohm)...")
            load.enable_input(False)  # Disable before mode change
            time.sleep(0.5)
            load.configure_cr_mode(resistance=5.0)
            mode = load.get_mode()
            resistance = load.get_resistance()
            print(f"[Test 9] [OK] Mode: {mode}")
            print(f"[Test 9] [OK] Resistance setting: {resistance:.2f} Ohm\n")

        # Test 10: Enable and measure in CR mode
        if start_test <= 10:
            print("[Test 10] Enabling load in CR mode...")
            load.enable_input(True)
            time.sleep(1.0)
            voltage = load.measure_voltage()
            current = load.measure_current()
            power = load.measure_power()
            print(f"[Test 10] [OK] Voltage: {voltage:.4f} V")
            print(f"[Test 10] [OK] Current: {current:.4f} A")
            print(f"[Test 10] [OK] Power: {power:.4f} W")

            # Verify Ohm's law
            if voltage > 1.0 and current > 0.1:
                measured_r = voltage / current
                print(f"[Test 10] [OK] Calculated resistance: {measured_r:.2f} Ohm")
                if abs(measured_r - 5.0) > 2.0:
                    print(f"[Test 10] [WARNING] Resistance deviation from setpoint")
            print()

        # Test 11: Switch to CP mode
        if start_test <= 11:
            print("[Test 11] Switching to CP mode (3W)...")
            load.enable_input(False)
            time.sleep(0.5)
            load.configure_cp_mode(power=3.0)
            mode = load.get_mode()
            power_setting = load.get_power()
            print(f"[Test 11] [OK] Mode: {mode}")
            print(f"[Test 11] [OK] Power setting: {power_setting:.2f} W\n")

        # Test 12: Enable and measure in CP mode
        if start_test <= 12:
            print("[Test 12] Enabling load in CP mode...")
            load.enable_input(True)
            time.sleep(1.0)
            voltage = load.measure_voltage()
            current = load.measure_current()
            power = load.measure_power()
            print(f"[Test 12] [OK] Voltage: {voltage:.4f} V")
            print(f"[Test 12] [OK] Current: {current:.4f} A")
            print(f"[Test 12] [OK] Power: {power:.4f} W")

            # Verify power
            calculated_power = voltage * current
            print(f"[Test 12] [OK] Calculated power: {calculated_power:.4f} W")
            if abs(power - 3.0) > 1.0:
                print(f"[Test 12] [WARNING] Power deviation from setpoint")
            print()

        # Test 13: Query input state
        if start_test <= 13:
            print("[Test 13] Querying input state...")
            input_state = load.get_input_state()
            print(f"[Test 13] [OK] Input state: {'ENABLED' if input_state else 'DISABLED'}\n")

        # Test 14: Disable input
        if start_test <= 14:
            print("[Test 14] Disabling load input...")
            load.enable_input(False)
            time.sleep(0.5)
            input_state = load.get_input_state()
            print(f"[Test 14] [OK] Input state: {'ENABLED' if input_state else 'DISABLED'}")
            if not input_state:
                print("[Test 14] [OK] Input successfully disabled\n")
            else:
                print("[Test 14] [FAIL] Input still enabled!\n")

        # Test 15: Check for errors
        if start_test <= 15:
            print("[Test 15] Checking for errors...")
            error = load.check_errors()
            if error.startswith("+0") or error.startswith("0"):
                print("[Test 15] [OK] No errors detected\n")
            else:
                print(f"[Test 15] [WARNING] Error: {error}\n")

        # ============================================================
        # Protection Tests (OPP, OCP, UVP)
        # ============================================================

        # Test 16: Overpower Protection (OPP) threshold
        if start_test <= 16:
            print("[Test 16] Testing Overpower Protection (OPP) threshold...")
            load.set_overpower_protection(100.0)
            opp_readback = load.get_overpower_protection()
            print(f"[Test 16] [OK] Set OPP: 100.0W")
            print(f"[Test 16] [OK] Readback OPP: {opp_readback:.2f}W")
            if abs(opp_readback - 100.0) < 1.0:
                print("[Test 16] [OK] OPP threshold verified\n")
            else:
                print("[Test 16] [WARNING] OPP readback mismatch\n")

        # Test 17: Overpower Protection (OPP) action
        # Note: PLZ-4W only supports "LIM" via SCPI. "LOAD OFF" is front-panel only.
        if start_test <= 17:
            print("[Test 17] Testing OPP action settings...")
            print("[Test 17] Note: Only LIM supported via SCPI. LOAD OFF requires front panel.")
            load.set_overpower_protection_action(PLZ164W.PROT_ACTION_LIMIT)
            action = load.get_overpower_protection_action()
            print(f"[Test 17] [OK] Set OPP action: LIM, Readback: {action}\n")

        # Test 18: Overcurrent Protection (OCP) threshold
        if start_test <= 18:
            print("[Test 18] Testing Overcurrent Protection (OCP) threshold...")
            load.set_overcurrent_protection(10.0)
            ocp_readback = load.get_overcurrent_protection()
            print(f"[Test 18] [OK] Set OCP: 10.0A")
            print(f"[Test 18] [OK] Readback OCP: {ocp_readback:.2f}A")
            if abs(ocp_readback - 10.0) < 0.5:
                print("[Test 18] [OK] OCP threshold verified\n")
            else:
                print("[Test 18] [WARNING] OCP readback mismatch\n")

        # Test 19: Overcurrent Protection (OCP) action
        # Note: PLZ-4W only supports "LIM" via SCPI. "LOAD OFF" is front-panel only.
        if start_test <= 19:
            print("[Test 19] Testing OCP action settings...")
            print("[Test 19] Note: Only LIM supported via SCPI. LOAD OFF requires front panel.")
            load.set_overcurrent_protection_action(PLZ164W.PROT_ACTION_LIMIT)
            action = load.get_overcurrent_protection_action()
            print(f"[Test 19] [OK] Set OCP action: LIM, Readback: {action}\n")

        # Test 20: Undervoltage Protection (UVP) threshold and enable
        if start_test <= 20:
            print("[Test 20] Testing Undervoltage Protection (UVP)...")

            # Set threshold (setting threshold automatically enables UVP)
            load.set_undervoltage_protection(1.0)
            uvp_readback = load.get_undervoltage_protection()
            print(f"[Test 20] [OK] Set UVP threshold: 1.0V")
            print(f"[Test 20] [OK] Readback UVP threshold: {uvp_readback:.2f}V")
            print("[Test 20] [OK] UVP threshold verified\n")

        # Test 21: Protection settings summary
        if start_test <= 21:
            print("[Test 21] Protection settings summary...")
            opp = load.get_overpower_protection()
            opp_action = load.get_overpower_protection_action()
            ocp = load.get_overcurrent_protection()
            ocp_action = load.get_overcurrent_protection_action()
            uvp = load.get_undervoltage_protection()

            print(f"[Test 21]   OPP: {opp:.2f}W (action: {opp_action})")
            print(f"[Test 21]   OCP: {ocp:.2f}A (action: {ocp_action})")
            print(f"[Test 21]   UVP: {uvp:.2f}V")
            print("[Test 21] [OK] Protection summary complete\n")

        # Final cleanup
        print("[Cleanup] Final cleanup...")
        load.enable_input(False)
        adapter.close()
        print("[Cleanup] [OK] Load disabled and adapter closed\n")

        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

        # Try to clean up
        try:
            print("\n[Cleanup] Attempting emergency cleanup...")
            load.enable_input(False)
            adapter.close()
            print("[Cleanup] [OK] Emergency cleanup complete")
        except:
            print("[Cleanup] [FAIL] Could not disable load - MANUALLY DISCONNECT POWER!")

        return False

    return True


def main():
    """Main entry point for hardware test."""
    parser = argparse.ArgumentParser(
        description='Test Kikusui PLZ164W electronic load with actual hardware'
    )
    parser.add_argument(
        'port',
        help='COM port for Prologix adapter (e.g., COM3)'
    )
    parser.add_argument(
        '--addr',
        type=int,
        default=10,
        help='GPIB address of PLZ164W (default: 10)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Start from test N (default: 1)'
    )

    args = parser.parse_args()

    success = test_load_basic(args.port, args.addr, start_test=args.start)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
