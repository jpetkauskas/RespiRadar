"""Reads the radar in a background thread and streams results to the dashboard over a websocket."""

from __future__ import annotations

import argparse
import asyncio
import threading
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from respiradar.breathing import AppState, BreathingPipeline, BreathingResult
from respiradar.sources import RadarConfig, radar_frames, simulated_frames

STATIC = Path(__file__).parent / "static"
PUSH_INTERVAL_S = 0.1

app = FastAPI()
latest: dict = {"status": "starting"}


def to_payload(result: BreathingResult) -> dict:
    """Shape a pipeline result for the browser dashboard."""
    presence = result.presence
    target = None
    if result.distances_being_analyzed is not None:
        low, high = result.distances_being_analyzed
        target = float(presence.distances_m[(low + high) // 2])

    events = []
    if result.app_state == AppState.APNEA:
        events.append(f"APNEA: no breathing for {result.quiet_s:.0f} s")
    if result.rate_bpm is not None and result.rate_bpm < 8:
        events.append("low breathing rate")
    if result.rate_bpm is not None and result.rate_bpm > 30:
        events.append("high breathing rate")

    return {
        "t": result.t,
        "app_state": result.app_state.value,
        "distances_m": presence.distances_m.tolist(),
        "range_profile": presence.score.tolist(),
        "target_m": target,
        "times": result.times.tolist(),
        "displacement_mm": result.displacement_mm.tolist(),
        "rate_bpm": result.rate_bpm,
        "breathing_ratio": result.breathing_ratio,
        "quiet_s": result.quiet_s,
        "events": events,
    }


def acquisition_loop(port: str | None) -> None:
    global latest
    config = RadarConfig()
    try:
        frames = radar_frames(port, config=config) if port else simulated_frames(config)
        pipeline = BreathingPipeline(config)
        for frame in frames:
            latest = {
                "status": "running",
                "source": port or "simulator",
                **to_payload(pipeline.process(frame)),
            }
    except Exception as e:
        traceback.print_exc()
        latest = {"status": f"error: {e}"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest)
            await asyncio.sleep(PUSH_INTERVAL_S)
    except WebSocketDisconnect:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="RespiRadar dashboard")
    parser.add_argument("--port", help="radar serial port, e.g. COM6 (omit to use the simulator)")
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args()

    threading.Thread(target=acquisition_loop, args=(args.port,), daemon=True).start()
    print(f"Dashboard: http://localhost:{args.http_port}")
    uvicorn.run(app, host="127.0.0.1", port=args.http_port, log_level="warning")
