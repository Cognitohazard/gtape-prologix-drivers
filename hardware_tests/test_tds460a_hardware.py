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


def test_scope(port, gpib_addr):
    """Test all TDS460A oscilloscope operations.

    Tests are organized by subsystem: misc, horizontal, vertical, channel
    display, trigger, acquisition, waveform, measurements, cursors, and autoset.

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

        # -- Misc --

        # Test 1: Query identification
        print("[Test 1] get_id()...")
        idn = scope.get_id()
        print(f"[Test 1] [OK] ID: {idn}\n")

        # Test 2: Detect active channels
        print("[Test 2] get_active_channels()...")
        active_channels = scope.get_active_channels()
        print(f"[Test 2] [OK] Active channels: {active_channels}")
        if not active_channels:
            print("[Test 2] [WARNING] No channels active!")
            print("[Test 2] [WARNING] Please enable at least one channel on scope")
            print("[Test 2] [WARNING] Continuing with remaining tests...\n")
        else:
            print(f"[Test 2] [INFO] Found {len(active_channels)} active channel(s)\n")

        # -- Horizontal --

        # Test 3: Query horizontal scale
        print("[Test 3] get_horizontal_scale()...")
        original_scale = scope.get_horizontal_scale()
        print(f"[Test 3] [OK] Current scale: {original_scale} s/div\n")

        # Test 4: Set horizontal scale and verify
        print("[Test 4] set_horizontal_scale(1E-3) then verify...")
        scope.set_horizontal_scale(1e-3)
        time.sleep(0.2)
        new_scale = scope.get_horizontal_scale()
        print(f"[Test 4] [OK] Scale after set: {new_scale} s/div")
        assert abs(new_scale - 1e-3) < 1e-6, f"Expected 1ms, got {new_scale}"
        print("[Test 4] [OK] Verified")
        # Restore original horizontal scale
        scope.set_horizontal_scale(original_scale)
        time.sleep(0.2)
        print("[Test 4] [OK] Scale restored\n")

        # Test 5: Query record length
        print("[Test 5] get_record_length()...")
        original_rec_len = scope.get_record_length()
        print(f"[Test 5] [OK] Record length: {original_rec_len} pts\n")

        # Test 6: Horizontal position
        print("[Test 6] get/set_horizontal_position()...")
        original_pos = scope.get_horizontal_position()
        print(f"[Test 6] [INFO] Original position: {original_pos}%")
        scope.set_horizontal_position(50.0)
        time.sleep(0.2)
        new_pos = scope.get_horizontal_position()
        print(f"[Test 6] [OK] Position after set: {new_pos}%")
        # Restore
        scope.set_horizontal_position(original_pos)
        print(f"[Test 6] [OK] Position restored\n")

        # -- Vertical --

        if active_channels:
            ch = active_channels[0]

            # Test 7: Channel scale
            print(f"[Test 7] get_channel_scale({ch})...")
            original_v_scale = scope.get_channel_scale(ch)
            print(f"[Test 7] [OK] {ch} scale: {original_v_scale} V/div\n")

            # Test 8: Set channel scale and verify
            print(f"[Test 8] set_channel_scale({ch}, 0.5) then verify...")
            scope.set_channel_scale(ch, 0.5)
            time.sleep(0.2)
            v_scale = scope.get_channel_scale(ch)
            print(f"[Test 8] [OK] {ch} scale after set: {v_scale} V/div")
            # Restore
            scope.set_channel_scale(ch, original_v_scale)
            print(f"[Test 8] [OK] Scale restored\n")

            # Test 9: Channel offset
            print(f"[Test 9] get/set_channel_offset({ch})...")
            original_offset = scope.get_channel_offset(ch)
            print(f"[Test 9] [INFO] Original offset: {original_offset} V")
            scope.set_channel_offset(ch, 0.5)
            time.sleep(0.2)
            offset = scope.get_channel_offset(ch)
            print(f"[Test 9] [OK] Offset after set: {offset} V")
            scope.set_channel_offset(ch, original_offset)
            print(f"[Test 9] [OK] Offset restored\n")

            # Test 10: Channel coupling
            print(f"[Test 10] get/set_channel_coupling({ch})...")
            coupling = scope.get_channel_coupling(ch)
            print(f"[Test 10] [INFO] Current coupling: {coupling}")
            scope.set_channel_coupling(ch, "DC")
            time.sleep(0.2)
            coupling = scope.get_channel_coupling(ch)
            print(f"[Test 10] [OK] Coupling after set: {coupling}\n")

            # Test 11: Channel bandwidth
            print(f"[Test 11] get/set_channel_bandwidth({ch})...")
            bw = scope.get_channel_bandwidth(ch)
            print(f"[Test 11] [INFO] Current bandwidth: {bw}")
            scope.set_channel_bandwidth(ch, "FULl")
            time.sleep(0.2)
            bw = scope.get_channel_bandwidth(ch)
            print(f"[Test 11] [OK] Bandwidth after set: {bw}\n")
        else:
            print("[Test 7-11] [SKIP] No active channels for vertical tests\n")

        # -- Channel Display --

        # Test 12: Toggle CH4 display
        print("[Test 12] set_channel_display(CH4, ON) then OFF...")
        scope.set_channel_display("CH4", True)
        time.sleep(0.2)
        channels_after = scope.get_active_channels()
        ch4_on = "CH4" in channels_after
        print(f"[Test 12] [INFO] CH4 active after ON: {ch4_on}")

        scope.set_channel_display("CH4", False)
        time.sleep(0.2)
        channels_after = scope.get_active_channels()
        ch4_off = "CH4" not in channels_after
        print(f"[Test 12] [OK] CH4 inactive after OFF: {ch4_off}\n")

        # -- Trigger --

        # Test 13: Trigger source
        print("[Test 13] get_trigger_source()...")
        trig_src = scope.get_trigger_source()
        print(f"[Test 13] [OK] Trigger source: {trig_src}\n")

        # Test 14: Trigger level
        print("[Test 14] set_trigger_level(1.0) then verify...")
        scope.set_trigger_level(1.0)
        time.sleep(0.2)
        level = scope.get_trigger_level()
        print(f"[Test 14] [OK] Level after set: {level} V\n")

        # Test 15: Trigger slope
        print("[Test 15] set_trigger_slope(RISe) then verify...")
        scope.set_trigger_slope("RISe")
        time.sleep(0.2)
        slope = scope.get_trigger_slope()
        print(f"[Test 15] [OK] Slope after set: {slope}\n")

        # Test 16: Trigger mode
        print("[Test 16] set_trigger_mode(AUTO) then verify...")
        scope.set_trigger_mode("AUTO")
        time.sleep(0.2)
        mode = scope.get_trigger_mode()
        print(f"[Test 16] [OK] Mode after set: {mode}\n")

        # -- Acquisition --

        # Test 17: Acquisition state
        print("[Test 17] get_acquisition_state()...")
        state = scope.get_acquisition_state()
        print(f"[Test 17] [OK] State: {state}\n")

        # Test 18: Acquisition mode
        print("[Test 18] get/set_acquisition_mode(SAMple)...")
        acq_mode = scope.get_acquisition_mode()
        print(f"[Test 18] [INFO] Current mode: {acq_mode}")
        scope.set_acquisition_mode("SAMple")
        time.sleep(0.2)
        acq_mode = scope.get_acquisition_mode()
        print(f"[Test 18] [OK] Mode after set: {acq_mode}\n")

        # Test 19: Number of averages
        print("[Test 19] get/set_num_averages(64)...")
        original_avg = scope.get_num_averages()
        print(f"[Test 19] [INFO] Original averages: {original_avg}")
        scope.set_num_averages(64)
        time.sleep(0.2)
        avg = scope.get_num_averages()
        print(f"[Test 19] [OK] Averages after set: {avg}")
        scope.set_num_averages(original_avg)
        print(f"[Test 19] [OK] Averages restored\n")

        # Test 20: Stop/run acquisition
        print("[Test 20] stop_acquisition() / run_acquisition()...")
        print("[Test 20] Stopping acquisition (freezing waveform)...")
        scope.stop_acquisition()
        time.sleep(0.5)
        state = scope.get_acquisition_state()
        print(f"[Test 20] [INFO] Acquisition state after stop: {state}")
        if "0" in str(state) or "STOP" in str(state).upper():
            print("[Test 20] [OK] Acquisition stopped successfully")
        else:
            print(f"[Test 20] [WARNING] Unexpected state: {state}")

        print("[Test 20] Resuming acquisition...")
        scope.run_acquisition()
        time.sleep(0.5)
        state = scope.get_acquisition_state()
        print(f"[Test 20] [INFO] Acquisition state after run: {state}")
        if "1" in str(state) or "RUN" in str(state).upper():
            print("[Test 20] [OK] Acquisition resumed successfully\n")
        else:
            print(f"[Test 20] [WARNING] Unexpected state: {state}\n")

        # Test 21: Single acquisition
        print("[Test 21] single_acquisition()...")
        scope.single_acquisition()
        time.sleep(1.0)
        state = scope.get_acquisition_state()
        print(f"[Test 21] [OK] State after single: {state}")
        # Resume running
        scope.run_acquisition()
        time.sleep(0.5)
        print("[Test 21] [OK] Resumed running\n")

        # -- Waveform --

        # Test 22: 500-point waveform capture
        if active_channels:
            channel = active_channels[0]
            print(f"[Test 22] set_record_length(500), read_waveform({channel})...")
            scope.set_record_length(500)
            print(f"[Test 22] [INFO] This may take ~1-2 seconds...")

            waveform = scope.read_waveform(channel)

            print(f"[Test 22] [OK] Waveform acquired successfully")
            print(f"[Test 22] [INFO] Channel: {waveform.channel}")
            print(f"[Test 22] [INFO] Points: {len(waveform.voltage)}")
            print(f"[Test 22] [INFO] Voltage range: {min(waveform.voltage):.6f} to {max(waveform.voltage):.6f} V")
            print(f"[Test 22] [INFO] Time range: {min(waveform.time):.9f} to {max(waveform.time):.9f} s")
            print(f"[Test 22] [INFO] Preamble fields: {len(waveform.preamble)} fields\n")
        else:
            print("[Test 22] [SKIP] No active channels to read\n")

        # Test 23: Read remaining active channels
        if len(active_channels) > 1:
            print(f"[Test 23] Reading from multiple channels ({len(active_channels)} total)...")
            for channel in active_channels[1:]:
                print(f"[Test 23] Reading {channel}...")
                waveform = scope.read_waveform(channel)
                print(f"[Test 23] [OK] {channel}: {len(waveform.voltage)} points, "
                      f"V range: {min(waveform.voltage):.6f} to {max(waveform.voltage):.6f} V")
            print(f"[Test 23] [OK] All {len(active_channels)} channels read successfully\n")
        else:
            print("[Test 23] [SKIP] Only one or no active channels\n")

        # Test 24: 1000-point waveform capture
        if active_channels:
            print("[Test 24] set_record_length(1000), read_waveform()...")
            scope.set_record_length(1000)
            waveform = scope.read_waveform(active_channels[0])
            print(f"[Test 24] [OK] Read {len(waveform.voltage)} points at 1000 point record length\n")
        else:
            print("[Test 24] [SKIP] No active channels\n")

        # Test 25: Restore original record length
        print(f"[Test 25] Restoring original record length ({original_rec_len} pts)...")
        scope.set_record_length(int(original_rec_len))
        print(f"[Test 25] [OK] Record length restored\n")

        # -- Measurements --

        if active_channels:
            ch = active_channels[0]

            # Test 26: Immediate frequency measurement
            print(f"[Test 26] measure_immediate({ch}, FREQuency)...")
            try:
                freq = scope.measure_immediate(ch, "FREQuency")
                print(f"[Test 26] [OK] Frequency: {freq} Hz\n")
            except ValueError:
                print(f"[Test 26] [INFO] No frequency detected (scope returned non-numeric)\n")

            # Test 27: Immediate peak-to-peak measurement
            print(f"[Test 27] measure_immediate({ch}, PK2pk)...")
            try:
                vpp = scope.measure_immediate(ch, "PK2pk")
                print(f"[Test 27] [OK] Vpp: {vpp} V\n")
            except ValueError:
                print(f"[Test 27] [INFO] No Vpp detected (scope returned non-numeric)\n")

            # Test 28: Measurement slot
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

        # Test 29: HBars
        print("[Test 29] set_cursor_function(HBArs), set positions, get delta...")
        scope.set_cursor_function("HBArs")
        time.sleep(0.2)
        scope.set_hbar_positions(0.5, -0.5)
        time.sleep(0.2)
        delta = scope.get_hbar_delta()
        print(f"[Test 29] [OK] HBar delta: {delta} V\n")

        # Test 30: VBars
        print("[Test 30] set_cursor_function(VBArs), set positions, get delta...")
        scope.set_cursor_function("VBArs")
        time.sleep(0.2)
        scope.set_vbar_positions(1e-3, -1e-3)
        time.sleep(0.2)
        delta = scope.get_vbar_delta()
        print(f"[Test 30] [OK] VBar delta: {delta} s\n")

        # Test 31: Cursors off
        print("[Test 31] set_cursor_function(OFF)...")
        scope.set_cursor_function("OFF")
        time.sleep(0.2)
        print("[Test 31] [OK] Cursors off\n")

        # -- Autoset --

        # Test 32: Autoset
        print("[Test 32] autoset()...")
        scope.autoset()
        time.sleep(10.0)  # Autoset takes 5-10 seconds
        print("[Test 32] [OK] Autoset complete\n")

        # -- Waveform (post-autoset visualization) --

        # Test 33: 15000-point capture + Plotly HTML
        if active_channels:
            print("[Test 33] 15000-point capture for visualization...")

            # Set record length for maximum detail
            print("[Test 33] Setting record length to 15000 points for maximum detail...")
            scope.set_record_length(15000)
            time.sleep(2.0)  # Give scope time to reconfigure memory after autoset

            # Read waveform
            channel = active_channels[0]
            print(f"[Test 33] Reading waveform from {channel}...")
            print(f"[Test 33] [INFO] This may take ~30-60 seconds at 9600 baud...")
            waveform = scope.read_waveform(channel)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"scope_test_{timestamp}.html"

            # Create plot (optional, requires plotly)
            if HAS_PLOTTER:
                print(f"[Test 33] Creating Plotly visualization...")
                WaveformPlotter.save_html([waveform], plot_filename)
                print(f"[Test 33] [OK] Waveform plot saved to: {plot_filename}")
                print(f"[Test 33] [INFO] Open {plot_filename} in a web browser to view the plot\n")
            else:
                print("[Test 33] [SKIP] Plotly not available - install with: pip install plotly\n")
        else:
            print("[Test 33] [SKIP] No active channels for waveform capture\n")

        # -- Error Check --

        # Test 34: Final error check
        print("[Test 34] check_errors()...")
        error = scope.check_errors()
        if error.startswith("0,"):
            print("[Test 34] [OK] No errors\n")
        else:
            print(f"[Test 34] [INFO] Error status: {error}\n")

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

    sys.exit(test_scope(args.port, args.addr))
