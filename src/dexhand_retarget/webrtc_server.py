from __future__ import annotations

import asyncio
import sys
import threading
import time
from ssl import SSLContext
from typing import Any, TextIO

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .state import SharedState


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=(
            "DexHandRetarget WebRTC receiver is running.\n"
            "POST SDP offers to /offer.\n"
        ),
        content_type="text/plain",
    )


async def handle_offer(request: web.Request) -> web.Response:
    state: SharedState = request.app["state"]
    pcs = request.app["pcs"]
    log_stream: TextIO = request.app["log_stream"]
    replay_writer = request.app["replay_writer"]
    remote = request.remote or "unknown"

    print(f"[signaling] POST /offer from {remote}", file=log_stream, flush=True)

    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    except Exception as exc:
        state.set_error(f"Invalid offer: {exc}")
        print(f"[signaling] Invalid offer from {remote}: {exc}", file=log_stream, flush=True)
        return web.json_response({"error": f"Invalid offer: {exc}"}, status=400)

    pc = RTCPeerConnection()
    pcs.add(pc)
    peer_id = f"peer-{id(pc):x}"
    state.set_peer_state("created")
    log_peer_states(peer_id, pc, log_stream)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        state.set_peer_state(pc.connectionState)
        print(f"[{peer_id}] Peer state: {pc.connectionState} | {transport_summary(pc)}", file=log_stream, flush=True)
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            pcs.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        state.set_channel_state(channel.label, "created")
        print(f"[{peer_id}] DataChannel received: {channel.label}", file=log_stream, flush=True)
        message_count = 0
        last_rx_log_at = time.monotonic()

        @channel.on("open")
        def on_open() -> None:
            state.set_channel_state(channel.label, "open")
            print(f"[{peer_id}] DataChannel open", file=log_stream, flush=True)

        @channel.on("close")
        def on_close() -> None:
            state.set_channel_state(channel.label, "closed")
            print(f"[{peer_id}] DataChannel closed", file=log_stream, flush=True)

        @channel.on("message")
        def on_message(message: Any) -> None:
            nonlocal message_count, last_rx_log_at
            frame = state.record_message(message)
            if frame is not None and replay_writer is not None:
                try:
                    replay_writer.record_frame(frame)
                except Exception as exc:
                    state.set_error(f"Replay save failed: {exc}")
                    print(f"[replay] Save failed: {exc}", file=log_stream, flush=True)
            message_count += 1
            now = time.monotonic()
            if now - last_rx_log_at >= 2.0:
                snapshot = state.snapshot()
                frame = snapshot["latest_frame"]
                source = "-" if frame is None else frame.source
                sequence = "-" if frame is None else frame.sequence
                print(
                    f"[{peer_id}] RX frames={message_count} "
                    f"latest={sequence} source={source} "
                    f"rx_fps={snapshot['rx_fps']:.1f}",
                    file=log_stream,
                    flush=True,
                )
                last_rx_log_at = now

    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as exc:
        state.set_error(f"WebRTC negotiation failed: {exc}")
        await pc.close()
        pcs.discard(pc)
        return web.json_response({"error": f"WebRTC negotiation failed: {exc}"}, status=500)

    print(f"[{peer_id}] SDP answer created", file=log_stream, flush=True)
    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


def log_peer_states(peer_id: str, pc: RTCPeerConnection, log_stream: TextIO) -> None:
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        print(f"[{peer_id}] ICE state: {pc.iceConnectionState} | {transport_summary(pc)}", file=log_stream, flush=True)

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        print(f"[{peer_id}] ICE gathering: {pc.iceGatheringState}", file=log_stream, flush=True)

    @pc.on("signalingstatechange")
    async def on_signalingstatechange() -> None:
        print(f"[{peer_id}] Signaling state: {pc.signalingState}", file=log_stream, flush=True)


def transport_summary(pc: RTCPeerConnection) -> str:
    sctp = pc.sctp
    if sctp is None:
        return "sctp=None"
    dtls = sctp.transport
    ice = dtls.transport
    return f"sctp={sctp.state}, dtls={dtls.state}, ice={ice.state}"


async def wait_for_stop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(0.2)


async def cleanup_app(app: web.Application, runner: web.AppRunner) -> None:
    pcs = list(app["pcs"])
    if pcs:
        await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
        app["pcs"].clear()
    await runner.cleanup()


def server_thread_main(
    host: str,
    port: int,
    ssl_context: SSLContext | None,
    state: SharedState,
    stop_event: threading.Event,
    ready_event: threading.Event,
    log_stream: TextIO | None = None,
    replay_writer: Any | None = None,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = web.Application()
    app["state"] = state
    app["pcs"] = set()
    app["log_stream"] = log_stream or sys.stdout
    app["replay_writer"] = replay_writer
    app.router.add_get("/", handle_index)
    app.router.add_post("/offer", handle_offer)
    runner = web.AppRunner(app)

    async def start() -> None:
        await runner.setup()
        site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
        await site.start()

    try:
        loop.run_until_complete(start())
    except Exception as exc:
        state.set_error(f"Server failed to start: {exc}")
        ready_event.set()
        loop.close()
        return

    ready_event.set()
    try:
        loop.run_until_complete(wait_for_stop(stop_event))
    finally:
        loop.run_until_complete(cleanup_app(app, runner))
        loop.close()
