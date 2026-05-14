from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import TextIO

from .network import create_ssl_context, print_urls
from .state import SharedState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive hand skeleton frames over WebRTC."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for signaling.")
    parser.add_argument("--port", type=int, help="Port for signaling. Defaults to 8080 for HTTP, 8443 for HTTPS.")
    parser.add_argument("--https", action="store_true", help="Serve signaling over HTTPS.")
    parser.add_argument("--cert-file", help="TLS certificate file for --https.")
    parser.add_argument("--key-file", help="TLS private key file for --https.")
    parser.add_argument(
        "--draw-axes",
        action="store_true",
        help="Draw wrist and palm orientation axes from received quaternions.",
    )
    parser.add_argument(
        "--retarget-dfq",
        action="store_true",
        help="Retarget one Quest hand to the Inspire DFQ MuJoCo model.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Connect to a real Inspire hand and send commands to it.",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        default="right",
        help="Selected hand for --retarget-dfq.",
    )
    parser.add_argument(
        "--dfq-model-path",
        type=Path,
        help="Override the Inspire DFQ MJCF model path.",
    )
    parser.add_argument(
        "--command-output",
        choices=("stdout", "off"),
        default="stdout",
        help="Where to emit DFQ actuator commands in --retarget-dfq mode.",
    )
    parser.add_argument(
        "--retarget-alpha",
        type=float,
        default=0.45,
        help="EMA smoothing alpha for DFQ commands. Lower values reduce latency.",
    )
    parser.add_argument(
        "--retarget-max-nfev",
        type=int,
        default=25,
        help="Maximum SciPy least_squares evaluations per retarget frame.",
    )
    parser.add_argument(
        "--retarget-max-step",
        type=float,
        default=0.16,
        help="Maximum per-actuator ctrl change per frame. Use 0 to disable rate limiting.",
    )
    parser.add_argument(
        "--retarget-release-max-step",
        type=float,
        default=0.12,
        help="Maximum non-thumb finger opening ctrl change per frame. Use 0 to match --retarget-max-step.",
    )
    parser.add_argument(
        "--retarget-feature-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing alpha for Quest palm-frame finger features. Use 0 to disable feature smoothing.",
    )
    parser.add_argument(
        "--retarget-feature-deadband",
        type=float,
        default=0.015,
        help="Deadband for Quest feature jitter: radians for directions/bends and normalized units for distances.",
    )
    parser.add_argument(
        "--live-log-interval",
        type=float,
        default=0.0,
        help="Print live DFQ retarget diagnostics every N seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-replay",
        type=Path,
        metavar="PATH",
        help="Save received WebRTC hand skeleton frames to a JSONL replay file.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        metavar="PATH",
        help="Play a saved replay file instead of waiting for a WebRTC client.",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier when replay records contain timing.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        help="Override replay timing with a fixed playback FPS.",
    )
    parser.add_argument(
        "--replay-loop",
        action="store_true",
        help="Loop replay playback until the viewer is closed.",
    )
    parser.add_argument(
        "--replay-headless",
        action="store_true",
        help="In --replay mode, process every frame without opening a viewer and exit.",
    )
    parser.add_argument(
        "--eval-replay",
        type=Path,
        metavar="PATH",
        help="Evaluate a saved replay against the DFQ retargeter and exit.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        metavar="PATH",
        help="Write the replay evaluation JSON report to this path.",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=1,
        help="Evaluate every Nth replay frame.",
    )
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        help="Stop reading the replay after this many frames.",
    )
    parser.add_argument(
        "--eval-no-details",
        action="store_true",
        help="Omit per-frame thumb details from the evaluation JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.port = args.port or (8443 if args.https else 8080)

    if args.save_replay is not None and args.replay is not None:
        print("--save-replay cannot be used together with --replay", file=sys.stderr)
        return 2
    if args.replay_headless and args.replay is None:
        print("--replay-headless requires --replay", file=sys.stderr)
        return 2
    if args.eval_replay is not None and (args.save_replay is not None or args.replay is not None):
        print("--eval-replay cannot be used together with --save-replay or --replay", file=sys.stderr)
        return 2

    log_stream = sys.stderr if args.retarget_dfq and args.command_output == "stdout" else sys.stdout

    if args.eval_replay is not None:
        return run_eval_replay(args, log_stream)

    if args.replay is not None and args.replay_headless:
        return run_headless_replay(args, log_stream)

    if args.replay is not None:
        return run_replay_viewer(args, log_stream)

    return run_live_server(args, log_stream)


def run_live_server(args: argparse.Namespace, log_stream: TextIO) -> int:
    try:
        ssl_context, temp_cert_dir = create_ssl_context(args, stream=log_stream)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        from .webrtc_server import server_thread_main
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    state = SharedState()
    stop_event = threading.Event()
    ready_event = threading.Event()
    replay_writer = None
    if args.save_replay is not None:
        try:
            from .replay import ReplayWriter

            replay_writer = ReplayWriter(args.save_replay, stream=log_stream)
        except OSError as exc:
            print(f"Failed to open replay file: {exc}", file=sys.stderr)
            if temp_cert_dir is not None:
                temp_cert_dir.cleanup()
            return 1

    server_thread = threading.Thread(
        target=server_thread_main,
        args=(
            args.host,
            args.port,
            ssl_context,
            state,
            stop_event,
            ready_event,
            log_stream,
            replay_writer,
        ),
        daemon=True,
    )
    server_thread.start()
    ready_event.wait(timeout=5.0)

    snapshot = state.snapshot()
    if snapshot["error"].startswith("Server failed"):
        print(snapshot["error"], file=sys.stderr)
        if replay_writer is not None:
            replay_writer.close()
        if temp_cert_dir is not None:
            temp_cert_dir.cleanup()
        return 1

    scheme = "https" if args.https else "http"
    print_urls(args.host, args.port, scheme, stream=log_stream)

    try:
        show_viewer(args, state, log_stream)
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server_thread.join(timeout=5.0)
        if replay_writer is not None:
            replay_writer.close()
        if temp_cert_dir is not None:
            temp_cert_dir.cleanup()

    return 0


def run_replay_viewer(args: argparse.Namespace, log_stream: TextIO) -> int:
    try:
        from .replay import ReplayError, ReplayPlayer
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    state = SharedState()
    stop_event = threading.Event()
    try:
        player = ReplayPlayer(
            state,
            args.replay,
            speed=args.replay_speed,
            fps=args.replay_fps,
            loop=args.replay_loop,
            log_stream=log_stream,
        )
    except (OSError, ReplayError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1

    def replay_thread_main() -> None:
        try:
            player.play(stop_event)
        except Exception as exc:  # pragma: no cover - defensive thread guard
            state.set_error(f"Replay failed: {exc}")
            print(f"[replay] Failed: {exc}", file=log_stream, flush=True)

    replay_thread = threading.Thread(target=replay_thread_main, daemon=True)
    replay_thread.start()
    try:
        show_viewer(args, state, log_stream)
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        replay_thread.join(timeout=5.0)
    return 0


def run_headless_replay(args: argparse.Namespace, log_stream: TextIO) -> int:
    try:
        from .replay import ReplayError, iter_replay_frames
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    if not args.retarget_dfq:
        count = 0
        try:
            for _frame in iter_replay_frames(args.replay):
                count += 1
        except (OSError, ReplayError) as exc:
            print(f"Replay failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"[replay] Parsed {count} frame(s) from {args.replay}",
            file=log_stream,
            flush=True,
        )
        return 0

    try:
        from .dfq_retarget import (
            CommandWriter,
            DfqModelAdapter,
            DfqRetargeter,
            default_dfq_model_path,
        )
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    model_path = args.dfq_model_path or default_dfq_model_path(args.hand)
    try:
        adapter = DfqModelAdapter(model_path, args.hand)
        retargeter = DfqRetargeter(
            adapter,
            args.hand,
            ema_alpha=args.retarget_alpha,
            max_nfev=args.retarget_max_nfev,
            max_ctrl_step=args.retarget_max_step,
            release_max_ctrl_step=args.retarget_release_max_step,
            feature_alpha=args.retarget_feature_alpha,
            feature_deadband=args.retarget_feature_deadband,
        )
    except Exception as exc:
        print(f"Failed to initialize DFQ retargeter: {exc}", file=sys.stderr)
        return 1

    writer = CommandWriter(args.command_output)
    print(
        f"[replay] Retargeting replay from {args.replay} with model={model_path}",
        file=log_stream,
        flush=True,
    )
    count = 0
    try:
        for frame in iter_replay_frames(args.replay):
            command = retargeter.command_for_frame(frame)
            adapter.apply_ctrl(command.ctrl)
            writer.write(command)
            count += 1
    except (OSError, ReplayError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    print(f"[replay] Finished {count} frame(s)", file=log_stream, flush=True)
    return 0


def run_eval_replay(args: argparse.Namespace, log_stream: TextIO) -> int:
    try:
        from .eval_retarget import (
            EvaluationConfig,
            EvaluationError,
            evaluate_replay,
            print_summary,
            write_report,
        )
        from .replay import ReplayError
    except ImportError as exc:
        print(
            "Missing Python dependency. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    config = EvaluationConfig(
        replay_path=args.eval_replay,
        hand=args.hand,
        model_path=args.dfq_model_path,
        ema_alpha=args.retarget_alpha,
        max_nfev=args.retarget_max_nfev,
        max_ctrl_step=args.retarget_max_step,
        release_max_ctrl_step=args.retarget_release_max_step,
        feature_alpha=args.retarget_feature_alpha,
        feature_deadband=args.retarget_feature_deadband,
        stride=args.eval_stride,
        max_frames=args.eval_max_frames,
        include_details=not args.eval_no_details,
    )
    try:
        report = evaluate_replay(config)
        if args.eval_output is not None:
            write_report(report, args.eval_output)
            print(f"[eval] Wrote report to {args.eval_output}", file=log_stream, flush=True)
        print_summary(report, stream=log_stream)
    except (EvaluationError, ReplayError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def show_viewer(args: argparse.Namespace, state: SharedState, log_stream: TextIO) -> None:
    if args.retarget_dfq:
        from .dfq_retarget import DfqRetargetRunner

        real_hand = None

        if args.real:
            from .inspire_hand import RealInspireHand

            real_hand = RealInspireHand()

        runner = DfqRetargetRunner(
            state,
            hand=args.hand,
            model_path=args.dfq_model_path,
            command_output=args.command_output,
            ema_alpha=args.retarget_alpha,
            max_nfev=args.retarget_max_nfev,
            live_log_interval=args.live_log_interval,
            max_ctrl_step=args.retarget_max_step,
            release_max_ctrl_step=args.retarget_release_max_step,
            feature_alpha=args.retarget_feature_alpha,
            feature_deadband=args.retarget_feature_deadband,
            log_stream=log_stream,
            real_hand=real_hand,
        )
        runner.show()
    else:
        from .plotter import SkeletonPlot

        plot = SkeletonPlot(state, draw_axes=args.draw_axes)
        plot.show()
