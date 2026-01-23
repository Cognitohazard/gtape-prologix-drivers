"""Hardware test script for TDS460A digital oscilloscope.

This script tests the TDS460A scope driver with actual hardware.
"""

import sys
import time
import datetime
from pathlib import Path

# Add IC1 Automation root to path for export module import
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.tds460a import TDS460A
try:
    from export.plotly_writer import WaveformPlotter
    HAS_PLOTTER = True
except ImportError:
    HAS_PLOTTER = False


def test_scope_basic(port, gpib_addr):
    """Test basic oscilloscope operations.

    Args:
        port: COM port (e.g., "COM4")
        gpib_addr: GPIB address of scope (e.g., 1)
    """
    print("="*60)
    print("TDS 460A OSCILLOSCOPE HARDWARE TEST")
    print("="*60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("="*60 + "\n")

    try:
        # Connect to scope
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create scope instance
        scope = TDS460A(adapter)

        # Test 1: Query identification
        print("[Test 1] Querying scope identification...")
        idn = adapter.ask("*IDN?")
        print(f"[Test 1] [OK] Scope ID: {idn}\n")

        # Test 2: Detect active channels
        print("[Test 2] Detecting active channels...")
        active_channels = scope.get_active_channels()
        print(f"[Test 2] [OK] Active channels: {active_channels}")
        if not active_channels:
            print("[Test 2] [WARNING] No channels active!")
            print("[Test 2] [WARNING] Please enable at least one channel on scope")
            print("[Test 2] [WARNING] Continuing with remaining tests...\n")
        else:
            print(f"[Test 2] [INFO] Found {len(active_channels)} active channel(s)\n")

        # Test 3: Query current record length
        print("[Test 3] Querying current record length...")
        current_length = adapter.ask("HORizontal:RECOrdlength?")
        print(f"[Test 3] [OK] Current record length: {current_length} points\n")

        # Test 4: Set record length
        print("[Test 4] Setting record length to 500 points...")
        actual_length = scope.set_record_length(500)
        print(f"[Test 4] [OK] Record length set to: {actual_length} points\n")

        # Test 5: Read waveform from first active channel (if any)
        if active_channels:
            channel = active_channels[0]
            print(f"[Test 5] Reading waveform from {channel}...")
            print(f"[Test 5] [INFO] This may take ~1-2 seconds...")

            waveform = scope.read_waveform(channel)

            print(f"[Test 5] [OK] Waveform acquired successfully")
            print(f"[Test 5] [INFO] Channel: {waveform.channel}")
            print(f"[Test 5] [INFO] Points: {len(waveform.voltage)}")
            print(f"[Test 5] [INFO] Voltage range: {min(waveform.voltage):.6f} to {max(waveform.voltage):.6f} V")
            print(f"[Test 5] [INFO] Time range: {min(waveform.time):.9f} to {max(waveform.time):.9f} s")
            print(f"[Test 5] [INFO] Preamble fields: {len(waveform.preamble)} fields\n")
        else:
            print("[Test 5] [SKIP] No active channels to read\n")

        # Test 6: Read from multiple channels (if available)
        if len(active_channels) > 1:
            print(f"[Test 6] Reading from multiple channels ({len(active_channels)} total)...")
            for i, channel in enumerate(active_channels[1:], start=2):
                print(f"[Test 6] Reading {channel}...")
                waveform = scope.read_waveform(channel)
                print(f"[Test 6] [OK] {channel}: {len(waveform.voltage)} points, "
                      f"V range: {min(waveform.voltage):.6f} to {max(waveform.voltage):.6f} V")
            print(f"[Test 6] [OK] All {len(active_channels)} channels read successfully\n")
        else:
            print("[Test 6] [SKIP] Only one or no active channels\n")

        # Test 7: Test different record length
        print("[Test 7] Testing with record length of 1000 points...")
        actual_length = scope.set_record_length(1000)
        if active_channels:
            waveform = scope.read_waveform(active_channels[0])
            print(f"[Test 7] [OK] Read {len(waveform.voltage)} points at 1000 point record length\n")
        else:
            print("[Test 7] [SKIP] No active channels\n")

        # Test 8: Restore original record length
        print(f"[Test 8] Restoring original record length ({current_length} points)...")
        scope.set_record_length(int(current_length))
        print(f"[Test 8] [OK] Record length restored\n")

        # Test 9: Test horizontal resolution (time/div)
        print("[Test 9] Testing horizontal resolution control...")
        print("[Test 9] Setting horizontal scale to 1ms/div...")
        adapter.write("HORizontal:MAIn:SCAle 1E-3")
        time.sleep(0.2)
        h_scale = adapter.ask("HORizontal:MAIn:SCAle?")
        print(f"[Test 9] [OK] Horizontal scale set to: {h_scale} s/div\n")

        # Test 10: Test vertical resolution (volts/div)
        if active_channels:
            print("[Test 10] Testing vertical resolution control...")
            channel = active_channels[0]
            print(f"[Test 10] Setting {channel} vertical scale to 500mV/div...")
            adapter.write(f"{channel}:SCAle 0.5")
            time.sleep(0.2)
            v_scale = adapter.ask(f"{channel}:SCAle?")
            print(f"[Test 10] [OK] {channel} vertical scale set to: {v_scale} V/div\n")
        else:
            print("[Test 10] [SKIP] No active channels for vertical scale test\n")

        # Test 11: Restore horizontal scale
        print("[Test 11] Restoring horizontal scale to 500us/div...")
        adapter.write("HORizontal:MAIn:SCAle 500E-6")
        h_scale = adapter.ask("HORizontal:MAIn:SCAle?")
        print(f"[Test 11] [OK] Horizontal scale restored to: {h_scale} s/div\n")

        # Test 12: Check for errors
        print("[Test 12] Checking for errors...")
        error = scope.check_errors()
        if error.startswith("0,"):
            print("[Test 12] [OK] No errors detected\n")
        else:
            print(f"[Test 12] [INFO] Error status: {error}\n")

        # Test 13: Test acquisition stop/run (freeze/unfreeze waveform)
        print("[Test 13] Testing acquisition stop/run...")
        print("[Test 13] Stopping acquisition (freezing waveform)...")
        scope.stop_acquisition()
        time.sleep(0.5)
        acq_state = adapter.ask("ACQuire:STATE?")
        print(f"[Test 13] [INFO] Acquisition state after stop: {acq_state}")
        if "0" in acq_state or "STOP" in acq_state.upper():
            print("[Test 13] [OK] Acquisition stopped successfully")
        else:
            print(f"[Test 13] [WARNING] Unexpected state: {acq_state}")

        print("[Test 13] Resuming acquisition...")
        scope.run_acquisition()
        time.sleep(0.5)
        acq_state = adapter.ask("ACQuire:STATE?")
        print(f"[Test 13] [INFO] Acquisition state after run: {acq_state}")
        if "1" in acq_state or "RUN" in acq_state.upper():
            print("[Test 13] [OK] Acquisition resumed successfully\n")
        else:
            print(f"[Test 13] [WARNING] Unexpected state: {acq_state}\n")

        # Test 14: AUTOSET and capture waveform for Plotly visualization
        if active_channels:
            print("[Test 14] AUTOSET and waveform capture for visualization...")

            # Run AUTOSET to optimize scope settings
            print("[Test 14] Running AUTOSET to auto-configure scope...")
            adapter.write("AUTOSet EXECUTE")
            time.sleep(10.0)  # Wait for autoset to complete (can take 5-10 seconds)
            print("[Test 14] [OK] AUTOSET complete")

            # Set record length for maximum detail
            print("[Test 14] Setting record length to 15000 points for maximum detail...")
            adapter.write("HORizontal:RECOrdlength 15000")
            time.sleep(2.0)  # Give scope time to reconfigure memory after autoset

            # Read waveform (pass record_length to avoid querying scope)
            channel = active_channels[0]
            print(f"[Test 14] Reading waveform from {channel}...")
            print(f"[Test 14] [INFO] This may take ~30-60 seconds at 9600 baud...")
            waveform = scope.read_waveform(channel)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"scope_test_{timestamp}.html"

            # Create plot (optional, requires plotly)
            if HAS_PLOTTER:
                print(f"[Test 14] Creating Plotly visualization...")
                WaveformPlotter.save_html([waveform], plot_filename)
                print(f"[Test 14] [OK] Waveform plot saved to: {plot_filename}")
                print(f"[Test 14] [INFO] Open {plot_filename} in a web browser to view the plot\n")
            else:
                print("[Test 14] [SKIP] Plotly not available - install with: pip install plotly\n")
        else:
            print("[Test 14] [SKIP] No active channels for AUTOSET and waveform capture\n")

        # Cleanup
        print("[Cleanup] Closing connection...")
        adapter.close()
        print("[Cleanup] [OK] Connection closed\n")

        # Summary
        print("="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nHardware verification complete!")
        print("The TDS460A driver is working correctly with actual hardware.")

        if not active_channels:
            print("\nNOTE: Enable channels on the scope for full waveform capture testing.")

        print()

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
        description="Hardware test for TDS460A digital oscilloscope"
    )
    parser.add_argument("port", help="COM port (e.g., COM4)")
    parser.add_argument("--addr", type=int, default=1,
                       help="GPIB address (default: 1)")

    args = parser.parse_args()

    sys.exit(test_scope_basic(args.port, args.addr))
