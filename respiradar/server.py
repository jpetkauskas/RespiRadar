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

from respiradar.breathing import BreathingPipeline
from respiradar.sources import FRAME_RATE_HZ, radar_frames, simulated_frames

STATIC = Path(__file__).parent / "static"
PUSH_INTERVAL_S = 0.1

app = FastAPI()
latest: dict = {"status": "starting"}


def acquisition_loop(port: str | None) -> None:
    global latest
    try:
        frames = radar_frames(port) if port else simulated_frames()
        pipeline = BreathingPipeline(FRAME_RATE_HZ)
        for frame in frames:
            latest = {"status": "running", "source": port or "simulator", **pipeline.process(frame)}
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
