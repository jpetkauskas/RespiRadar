# Nishant Vikramaditya, Vishnu Mangipudi, and Justinas Petkauskas
RespiRadar: HackMIT 2026 Project, Healthcare Track

## Quick start

Requires [uv](https://docs.astral.sh/uv/). The XM125 must be running Acconeer's **Exploration Server** firmware
(the same firmware the Acconeer Exploration Tool GUI uses). Close the Exploration Tool first, since only one
program can hold the COM port.

```
uv sync
uv run python -m respiradar              # simulator, no hardware needed
uv run python -m respiradar --port COM6  # real radar
```

Then open http://localhost:8000.

## Layout

- `respiradar/sources.py`: sensor config, XM125 reader (`acconeer.exptool.a121.Client`), simulator
- `respiradar/breathing.py`: pipeline IQ -> displacement -> rate -> alerts. The stages marked PLACEHOLDER are naive and meant to be replaced.
- `respiradar/server.py`: reads the radar in a background thread and pushes JSON to the browser over `/ws` every 100 ms
- `respiradar/static/index.html`: dashboard
RespiRadar: HackMIT 2026 Project, healthcare track

Board: SparkFun Pulsed Coherent Radar Sensor - Acconeer XM125 (sparkfun XM125 A121 Radar Breakout)
