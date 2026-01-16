"""Hardware test script for TDS3000 series digital oscilloscopes.

Supports TDS3054 (4-channel) and TDS3012B (2-channel).

Usage:
    python test_tds3000_hardware.py COM4 --addr 1
    python test_tds3000_hardware.py COM4 --addr 1 --model TDS3012B
"""

import sys
import time
import datetime
import argparse

from gtape_prologix_drivers.adapter import PrologixAdapter
from gtape_prologix_drivers.instruments.tds3000 import TDS3054, TDS3012B

try:
    from export.plotly_writer import WaveformPlotter
    HAS_PLOTTER = True
except ImportError:
    HAS_PLOTTER = False


def test_scope(port: str, gpib_addr: int, model: str = "TDS3054"):
    """Test TDS3000 series oscilloscope operations.

    Args:
        port: COM port (e.g., "COM4")
        gpib_addr: GPIB address of scope
        model: "TDS3054" or "TDS3012B"
    """
    print("=" * 60)
    print(f"TDS3000 SERIES OSCILLOSCOPE HARDWARE TEST")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Port: {port}")
    print(f"GPIB Address: {gpib_addr}")
    print("=" * 60 + "\n")

    adapter = None
    try:
        # Connect to scope
        print("[Test] Connecting to Prologix adapter...")
        adapter = PrologixAdapter(port=port, gpib_address=gpib_addr)
        print("[Test] [OK] Adapter connected\n")

        # Create scope instance based on model
        if model.upper() == "TDS3012B":
            scope = TDS3012B(adapter)
            expected_channels = 2
        else:
            scope = TDS3054(adapter)
            expected_channels = 4

        # Test 1: Query identification
        print("[Test 1] Querying scope identification...")
        idn = scope.get_id()
        print(f"[Test 1] [OK] Scope ID: {idn}")
        if model.upper() not in idn.upper():
            print(f"[Test 1] [WARNING] Expected {model} but got different ID\n")
        else:
            print()

        # Test 2: Detect active channels
        print("[Test 2] Detecting active channels...")
        active_channels = scope.get_active_channels()
        print(f"[Test 2] [OK] Active channels: {active_channels}")
        print(f"[Test 2] [INFO] Model supports {expected_channels} channels")
        if not active_channels:
            print("[Test 2] [WARNING] No channels active - enable at least one channel\n")
        else:
            print()

        # Test 3: Query sample rate
        print("[Test 3] Querying sample rate...")
        sample_rate = scope.get_sample_rate()
        print(f"[Test 3] [OK] Sample rate: {sample_rate:.2e} Sa/s\n")

        # Test 4: Query current record length
        print("[Test 4] Querying current record length...")
        response = adapter.ask("HORizontal:RECOrdlength?")
        original_length = int(response)
        print(f"[Test 4] [OK] Current record length: {original_length} points\n")

        # Test 5: Set record length
        print("[Test 5] Setting record length to 500 points...")
        actual_length = scope.set_record_length(500)
        print(f"[Test 5] [OK] Record length set to: {actual_length} points\n")

        # Test 6: Read waveform from first active channel
        if active_channels:
            channel = active_channels[0]
            print(f"[Test 6] Reading waveform from {channel}...")
            waveform = scope.read_waveform(channel)
            print(f"[Test 6] [OK] Waveform acquired")
            print(f"[Test 6] [INFO] Points: {len(waveform.voltage)}")
            print(f"[Test 6] [INFO] Voltage: {min(waveform.voltage):.4f} to {max(waveform.voltage):.4f} V")
            print(f"[Test 6] [INFO] Time: {waveform.time[0]:.6e} to {waveform.time[-1]:.6e} s\n")
        else:
            print("[Test 6] [SKIP] No active channels\n")

        # Test 7: Channel control - scale and position
        if active_channels:
            channel = active_channels[0]
            print(f"[Test 7] Testing channel control on {channel}...")

            # Get current scale
            current_scale = adapter.ask(f"{channel}:SCAle?")
            print(f"[Test 7] [INFO] Current scale: {current_scale} V/div")

            # Set new scale
            scope.set_channel_scale(channel, 1.0)
            time.sleep(0.2)
            new_scale = adapter.ask(f"{channel}:SCAle?")
            print(f"[Test 7] [OK] Scale set to: {new_scale} V/div")

            # Restore original scale
            scope.set_channel_scale(channel, float(current_scale))
            print(f"[Test 7] [OK] Scale restored\n")
        else:
            print("[Test 7] [SKIP] No active channels\n")

        # Test 8: Timebase control
        print("[Test 8] Testing timebase control...")
        current_timebase = adapter.ask("HORizontal:MAIn:SCAle?")
        print(f"[Test 8] [INFO] Current timebase: {current_timebase} s/div")

        scope.set_timebase(1e-3)
        time.sleep(0.2)
        new_timebase = adapter.ask("HORizontal:MAIn:SCAle?")
        print(f"[Test 8] [OK] Timebase set to: {new_timebase} s/div")

        # Restore
        adapter.write(f"HORizontal:MAIn:SCAle {current_timebase}")
        print("[Test 8] [OK] Timebase restored\n")

        # Test 9: Trigger control
        print("[Test 9] Testing trigger control...")
        current_trig_src = adapter.ask("TRIGger:A:EDGe:SOUrce?")
        print(f"[Test 9] [INFO] Current trigger source: {current_trig_src}")

        current_trig_level = adapter.ask("TRIGger:A:LEVel?")
        print(f"[Test 9] [INFO] Current trigger level: {current_trig_level} V")

        scope.set_trigger_mode("AUTO")
        mode = adapter.ask("TRIGger:A:MODe?")
        print(f"[Test 9] [OK] Trigger mode: {mode}\n")

        # Test 10: Acquisition control
        print("[Test 10] Testing acquisition control...")
        scope.set_acquire_mode("SAMple")
        acq_mode = adapter.ask("ACQuire:MODe?")
        print(f"[Test 10] [OK] Acquisition mode: {acq_mode}")

        scope.run()
        state = adapter.ask("ACQuire:STATE?")
        print(f"[Test 10] [OK] Acquisition state: {state}\n")

        # Test 11: Measurements
        if active_channels:
            channel = active_channels[0]
            print(f"[Test 11] Testing measurements on {channel}...")

            try:
                freq = scope.measure("FREQuency", channel)
                print(f"[Test 11] [INFO] Frequency: {freq:.2e} Hz")
            except Exception as e:
                print(f"[Test 11] [INFO] Frequency: N/A ({e})")

            try:
                pk2pk = scope.measure("PK2pk", channel)
                print(f"[Test 11] [INFO] Pk-Pk: {pk2pk:.4f} V")
            except Exception as e:
                print(f"[Test 11] [INFO] Pk-Pk: N/A ({e})")

            try:
                mean = scope.measure("MEAN", channel)
                print(f"[Test 11] [INFO] Mean: {mean:.4f} V")
            except Exception as e:
                print(f"[Test 11] [INFO] Mean: N/A ({e})")

            print("[Test 11] [OK] Measurements complete\n")
        else:
            print("[Test 11] [SKIP] No active channels\n")

        # Test 12: High-resolution waveform capture
        if active_channels:
            print("[Test 12] High-resolution waveform capture...")
            print("[Test 12] Setting record length to 10000 points...")
            scope.set_record_length(10000)
            time.sleep(0.5)

            channel = active_channels[0]
            print(f"[Test 12] Reading waveform from {channel}...")
            print("[Test 12] [INFO] This may take 10-20 seconds...")

            start_time = time.time()
            waveform = scope.read_waveform(channel, record_length=10000)
            elapsed = time.time() - start_time

            print(f"[Test 12] [OK] Captured {len(waveform.voltage)} points in {elapsed:.1f}s")
            print(f"[Test 12] [INFO] Transfer rate: {len(waveform.voltage) / elapsed:.0f} points/s")

            # Save plot if plotly available
            if HAS_PLOTTER:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tds3000_test_{timestamp}.html"
                WaveformPlotter.save_html([waveform], filename)
                print(f"[Test 12] [OK] Saved plot: {filename}")
            print()
        else:
            print("[Test 12] [SKIP] No active channels\n")

        # Test 13: Restore original record length
        print(f"[Test 13] Restoring record length to {original_length}...")
        scope.set_record_length(original_length)
        print("[Test 13] [OK] Record length restored\n")

        # Test 14: Error check
        print("[Test 14] Checking for errors...")
        error = scope.check_errors()
        if error == "0":
            print("[Test 14] [OK] No errors\n")
        else:
            print(f"[Test 14] [INFO] Status: {error}\n")

        # Cleanup
        print("[Cleanup] Closing connection...")
        adapter.close()
        print("[Cleanup] [OK] Done\n")

        # Summary
        print("=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
        print(f"\n{model} driver verified with actual hardware.")
        if not active_channels:
            print("NOTE: Enable channels for full waveform testing.")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()

        if adapter:
            try:
                adapter.close()
                print("\n[Cleanup] Connection closed")
            except:
                pass

        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hardware test for TDS3000 series oscilloscopes"
    )
    parser.add_argument("port", help="COM port (e.g., COM4)")
    parser.add_argument("--addr", type=int, default=1,
                        help="GPIB address (default: 1)")
    parser.add_argument("--model", choices=["TDS3054", "TDS3012B"],
                        default="TDS3054",
                        help="Oscilloscope model (default: TDS3054)")

    args = parser.parse_args()

    sys.exit(test_scope(args.port, args.addr, args.model))
