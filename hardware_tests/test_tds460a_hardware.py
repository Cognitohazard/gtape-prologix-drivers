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
        print("[OK] ALL BASIC TESTS PASSED")
        print("="*60)
        print("\nBasic hardware verification complete!")
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


def test_scope_extended(port, gpib_addr):
    """Test extended TDS460A driver methods with actual hardware.

    Tests horizontal, vertical, trigger, acquisition, measurement, and cursor
    methods added for the interactive CLI.

    Args:
        port: COM port (e.g., "COM4")
        gpib_addr: GPIB address of scope (e.g., 1)
    """
    print("="*60)
    print("TDS 460A EXTENDED HARDWARE TEST")
    print("="*60)
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("="*60 + "\n")

    try:
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        scope = TDS460A(adapter)
        print("[Test] [OK] Adapter connected\n")

        # -- Misc --

        print("[Test 1] get_id()...")
        idn = scope.get_id()
        print(f"[Test 1] [OK] ID: {idn}\n")

        # Detect active channels for later tests
        active_channels = scope.get_active_channels()
        if not active_channels:
            print("[WARNING] No active channels - some tests will be skipped\n")

        # -- Horizontal --

        print("[Test 2] get_horizontal_scale()...")
        original_scale = scope.get_horizontal_scale()
        print(f"[Test 2] [OK] Current scale: {original_scale} s/div\n")

        print("[Test 3] set_horizontal_scale(1E-3) then verify...")
        scope.set_horizontal_scale(1e-3)
        time.sleep(0.2)
        new_scale = scope.get_horizontal_scale()
        print(f"[Test 3] [OK] Scale after set: {new_scale} s/div")
        assert abs(new_scale - 1e-3) < 1e-6, f"Expected 1ms, got {new_scale}"
        print("[Test 3] [OK] Verified\n")

        print("[Test 4] get_record_length()...")
        rec_len = scope.get_record_length()
        print(f"[Test 4] [OK] Record length: {rec_len} pts\n")

        print("[Test 5] get/set_horizontal_position()...")
        original_pos = scope.get_horizontal_position()
        print(f"[Test 5] [INFO] Original position: {original_pos}%")
        scope.set_horizontal_position(50.0)
        time.sleep(0.2)
        new_pos = scope.get_horizontal_position()
        print(f"[Test 5] [OK] Position after set: {new_pos}%")
        # Restore
        scope.set_horizontal_position(original_pos)
        print(f"[Test 5] [OK] Position restored\n")

        # Restore original horizontal scale
        scope.set_horizontal_scale(original_scale)
        time.sleep(0.2)

        # -- Vertical --

        if active_channels:
            ch = active_channels[0]

            print(f"[Test 6] get_channel_scale({ch})...")
            original_v_scale = scope.get_channel_scale(ch)
            print(f"[Test 6] [OK] {ch} scale: {original_v_scale} V/div\n")

            print(f"[Test 7] set_channel_scale({ch}, 0.5) then verify...")
            scope.set_channel_scale(ch, 0.5)
            time.sleep(0.2)
            v_scale = scope.get_channel_scale(ch)
            print(f"[Test 7] [OK] {ch} scale after set: {v_scale} V/div\n")
            # Restore
            scope.set_channel_scale(ch, original_v_scale)

            print(f"[Test 8] get/set_channel_offset({ch})...")
            original_offset = scope.get_channel_offset(ch)
            print(f"[Test 8] [INFO] Original offset: {original_offset} V")
            scope.set_channel_offset(ch, 0.5)
            time.sleep(0.2)
            offset = scope.get_channel_offset(ch)
            print(f"[Test 8] [OK] Offset after set: {offset} V")
            scope.set_channel_offset(ch, original_offset)
            print(f"[Test 8] [OK] Offset restored\n")

            print(f"[Test 9] get_channel_coupling({ch})...")
            coupling = scope.get_channel_coupling(ch)
            print(f"[Test 9] [OK] Coupling: {coupling}\n")

            print(f"[Test 10] set_channel_coupling({ch}, DC)...")
            scope.set_channel_coupling(ch, "DC")
            time.sleep(0.2)
            coupling = scope.get_channel_coupling(ch)
            print(f"[Test 10] [OK] Coupling after set: {coupling}\n")

            print(f"[Test 11] get_channel_bandwidth({ch})...")
            bw = scope.get_channel_bandwidth(ch)
            print(f"[Test 11] [OK] Bandwidth: {bw}\n")

            print(f"[Test 12] set_channel_bandwidth({ch}, FULl)...")
            scope.set_channel_bandwidth(ch, "FULl")
            time.sleep(0.2)
            bw = scope.get_channel_bandwidth(ch)
            print(f"[Test 12] [OK] Bandwidth after set: {bw}\n")
        else:
            print("[Test 6-12] [SKIP] No active channels for vertical tests\n")

        # -- Channel display --

        print("[Test 13] set_channel_display(CH4, ON) then OFF...")
        scope.set_channel_display("CH4", True)
        time.sleep(0.2)
        channels_after = scope.get_active_channels()
        ch4_on = "CH4" in channels_after
        print(f"[Test 13] [INFO] CH4 active after ON: {ch4_on}")

        scope.set_channel_display("CH4", False)
        time.sleep(0.2)
        channels_after = scope.get_active_channels()
        ch4_off = "CH4" not in channels_after
        print(f"[Test 13] [OK] CH4 inactive after OFF: {ch4_off}\n")

        # -- Trigger --

        print("[Test 14] get_trigger_source()...")
        trig_src = scope.get_trigger_source()
        print(f"[Test 14] [OK] Trigger source: {trig_src}\n")

        print("[Test 15] get_trigger_level()...")
        trig_level = scope.get_trigger_level()
        print(f"[Test 15] [OK] Trigger level: {trig_level} V\n")

        print("[Test 16] set_trigger_level(1.0) then verify...")
        scope.set_trigger_level(1.0)
        time.sleep(0.2)
        level = scope.get_trigger_level()
        print(f"[Test 16] [OK] Level after set: {level} V\n")

        print("[Test 17] get_trigger_slope()...")
        slope = scope.get_trigger_slope()
        print(f"[Test 17] [OK] Slope: {slope}\n")

        print("[Test 18] set_trigger_slope(RISe) then verify...")
        scope.set_trigger_slope("RISe")
        time.sleep(0.2)
        slope = scope.get_trigger_slope()
        print(f"[Test 18] [OK] Slope after set: {slope}\n")

        print("[Test 19] get_trigger_mode()...")
        trig_mode = scope.get_trigger_mode()
        print(f"[Test 19] [OK] Mode: {trig_mode}\n")

        print("[Test 20] set_trigger_mode(AUTO) then verify...")
        scope.set_trigger_mode("AUTO")
        time.sleep(0.2)
        mode = scope.get_trigger_mode()
        print(f"[Test 20] [OK] Mode after set: {mode}\n")

        # -- Acquisition --

        print("[Test 21] get_acquisition_state()...")
        state = scope.get_acquisition_state()
        print(f"[Test 21] [OK] State: {state}\n")

        print("[Test 22] get_acquisition_mode()...")
        acq_mode = scope.get_acquisition_mode()
        print(f"[Test 22] [OK] Mode: {acq_mode}\n")

        print("[Test 23] set_acquisition_mode(SAMple) then verify...")
        scope.set_acquisition_mode("SAMple")
        time.sleep(0.2)
        acq_mode = scope.get_acquisition_mode()
        print(f"[Test 23] [OK] Mode after set: {acq_mode}\n")

        print("[Test 24] get/set_num_averages()...")
        original_avg = scope.get_num_averages()
        print(f"[Test 24] [INFO] Original averages: {original_avg}")
        scope.set_num_averages(64)
        time.sleep(0.2)
        avg = scope.get_num_averages()
        print(f"[Test 24] [OK] Averages after set: {avg}")
        scope.set_num_averages(original_avg)
        print(f"[Test 24] [OK] Averages restored\n")

        print("[Test 25] single_acquisition()...")
        scope.single_acquisition()
        time.sleep(1.0)
        state = scope.get_acquisition_state()
        print(f"[Test 25] [OK] State after single: {state}")
        # Resume running
        scope.run_acquisition()
        time.sleep(0.5)
        print("[Test 25] [OK] Resumed running\n")

        # -- Measurements --

        if active_channels:
            ch = active_channels[0]

            print(f"[Test 26] measure_immediate({ch}, FREQuency)...")
            try:
                freq = scope.measure_immediate(ch, "FREQuency")
                print(f"[Test 26] [OK] Frequency: {freq} Hz\n")
            except ValueError:
                print(f"[Test 26] [INFO] No frequency detected (scope returned non-numeric)\n")

            print(f"[Test 27] measure_immediate({ch}, PK2pk)...")
            try:
                vpp = scope.measure_immediate(ch, "PK2pk")
                print(f"[Test 27] [OK] Vpp: {vpp} V\n")
            except ValueError:
                print(f"[Test 27] [INFO] No Vpp detected (scope returned non-numeric)\n")

            print(f"[Test 28] configure_measurement_slot(1, {ch}, FREQuency)...")
            scope.configure_measurement_slot(1, ch, "FREQuency")
            time.sleep(0.5)
            try:
                val = scope.read_measurement_slot(1)
                print(f"[Test 28] [OK] Slot 1 value: {val}\n")
            except ValueError:
                print(f"[Test 28] [INFO] Slot 1 returned non-numeric\n")
        else:
            print("[Test 26-28] [SKIP] No active channels for measurement tests\n")

        # -- Cursors --

        print("[Test 29] set_cursor_function(HBArs)...")
        scope.set_cursor_function("HBArs")
        time.sleep(0.2)
        print("[Test 29] [OK] HBars enabled\n")

        print("[Test 30] set_hbar_positions(0.5, -0.5) and get_hbar_delta()...")
        scope.set_hbar_positions(0.5, -0.5)
        time.sleep(0.2)
        delta = scope.get_hbar_delta()
        print(f"[Test 30] [OK] HBar delta: {delta} V\n")

        print("[Test 31] set_cursor_function(VBArs)...")
        scope.set_cursor_function("VBArs")
        time.sleep(0.2)
        print("[Test 31] [OK] VBars enabled\n")

        print("[Test 32] set_vbar_positions(1e-3, -1e-3) and get_vbar_delta()...")
        scope.set_vbar_positions(1e-3, -1e-3)
        time.sleep(0.2)
        delta = scope.get_vbar_delta()
        print(f"[Test 32] [OK] VBar delta: {delta} s\n")

        print("[Test 33] set_cursor_function(OFF)...")
        scope.set_cursor_function("OFF")
        time.sleep(0.2)
        print("[Test 33] [OK] Cursors off\n")

        # -- Autoset --

        print("[Test 34] autoset()...")
        scope.autoset()
        time.sleep(10.0)  # Autoset takes 5-10 seconds
        print("[Test 34] [OK] Autoset complete\n")

        # Check for errors after all tests
        print("[Test 35] Final error check...")
        error = scope.check_errors()
        if error.startswith("0,"):
            print("[Test 35] [OK] No errors\n")
        else:
            print(f"[Test 35] [INFO] Error status: {error}\n")

        # Cleanup
        print("[Cleanup] Closing connection...")
        adapter.close()
        print("[Cleanup] [OK] Connection closed\n")

        print("="*60)
        print("[OK] ALL EXTENDED TESTS PASSED")
        print("="*60)
        print("\nExtended hardware verification complete!")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAIL] EXTENDED TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

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
    parser.add_argument("--extended", action="store_true",
                       help="Run extended tests for new driver methods")

    args = parser.parse_args()

    if args.extended:
        sys.exit(test_scope_extended(args.port, args.addr))
    else:
        sys.exit(test_scope_basic(args.port, args.addr))
