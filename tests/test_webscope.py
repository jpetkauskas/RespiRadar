"""What the scope server puts on the wire.

The browser draws these series on top of one another - the alarm shading behind the wave, the
hold shading behind the alarm - so the payload's shape is load-bearing, not cosmetic. A
mismatch does not crash anything; it just draws the alarm starting somewhere it did not.
"""

import itertools
import json
import math

import numpy as np
import pytest

from respiradar.sources import RadarConfig, simulated_frames
from respiradar.webscope import SERIES_POINTS, ScopeFeed, _round, _sample_indices


def _fed(seconds: int = 40, **kwargs) -> ScopeFeed:
    """A feed with `seconds` of simulated data already pushed through it, synchronously."""
    config = RadarConfig(sweeps_per_frame=8)
    feed = ScopeFeed(iter(()), config, **kwargs)
    frames = simulated_frames(config, breaths_per_min=14.0, realtime=False)
    for frame in itertools.islice(frames, int(seconds * config.frame_rate)):
        feed._ingest(frame)
    feed.worker.stop()
    return feed


# -- decimation ---------------------------------------------------------


def test_every_series_is_sampled_at_the_same_points():
    """One index set for all of them. Decimating each independently is how they drift."""
    feed = _fed()
    series = feed.snapshot()["series"]
    lengths = {name: len(values) for name, values in series.items()}
    assert len(set(lengths.values())) == 1, lengths


def test_decimation_keeps_the_wave_rather_than_aliasing_it():
    """A plain stride can sample a breath at its zero crossings and report a flat line."""
    fs, bpm = 20.0, 14.0
    t = np.arange(0, 30, 1 / fs)
    wave = 2.0 * np.sin(2 * np.pi * bpm / 60 * t)

    kept = wave[_sample_indices(wave, 64)]
    strided = wave[:: len(wave) // 64]
    assert np.abs(kept).max() > 0.9 * np.abs(wave).max()
    assert np.abs(kept).max() >= np.abs(strided).max()


def test_sample_indices_are_in_range_and_ordered():
    indices = _sample_indices(np.random.default_rng(0).standard_normal(1000), 64)
    assert len(indices) == 64
    assert indices.min() >= 0 and indices.max() < 1000
    assert np.all(np.diff(indices) > 0)


def test_a_short_history_is_sent_whole():
    values = np.arange(10.0)
    assert list(_sample_indices(values, SERIES_POINTS)) == list(range(10))


# -- the payload --------------------------------------------------------


def test_the_payload_is_real_json():
    """`json.dumps` writes NaN and Infinity as bare tokens, which `JSON.parse` rejects - the
    page goes blank and nothing says why."""
    feed = _fed()
    text = json.dumps(feed.snapshot(), allow_nan=False)  # raises on NaN/Infinity
    assert json.loads(text)


def test_no_non_finite_values_reach_the_client():
    feed = _fed()
    payload = feed.snapshot()

    def walk(node, path="payload"):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, float):
            assert math.isfinite(node), path

    walk(payload)


def test_round_survives_an_empty_or_broken_window():
    assert _round([]) == []
    assert _round([float("nan"), float("inf"), 1.0]) == [None, None, 1.0]


def test_the_payload_carries_every_panel():
    feed = _fed()
    payload = feed.snapshot()
    for key in ("profile", "bin_motion", "iq", "series", "spectrum", "distances_m",
                "chest_bin", "bpm", "present", "state", "holds"):
        assert key in payload, key
    assert len(payload["profile"]) == len(payload["distances_m"])
    assert len(payload["bin_motion"]) == len(payload["distances_m"])
    assert 0 <= payload["chest_bin"] < len(payload["distances_m"])


def test_the_payload_stays_small_enough_to_push_five_times_a_second():
    """The whole reason for decimating. A full 20 Hz history would be tens of MB a minute."""
    feed = _fed()
    assert len(json.dumps(feed.snapshot())) < 32_000


def test_a_starting_feed_reports_status_instead_of_half_a_payload():
    feed = ScopeFeed(iter(()), RadarConfig(sweeps_per_frame=8))
    feed.worker.stop()
    payload = feed.snapshot()
    assert "status" in payload and "series" not in payload


# -- what it says about the world ---------------------------------------


def test_a_breathing_simulator_is_reported_as_breathing():
    feed = _fed(seconds=60)
    payload = feed.snapshot()
    assert payload["present"] is True
    assert payload["bpm"] == pytest.approx(14.0, abs=2.0)
    assert payload["state"] in ("breathing", "APNEA")


def test_no_breathing_rate_is_invented_for_an_empty_room():
    """The largest bin in a band of noise is still the largest bin. Reporting it as a
    breathing rate is the failure this guards."""
    config = RadarConfig(sweeps_per_frame=8)
    feed = ScopeFeed(iter(()), config)
    frames = simulated_frames(config, breaths_per_min=None, realtime=False)
    for frame in itertools.islice(frames, int(60 * config.frame_rate)):
        feed._ingest(frame)
    feed.worker.stop()

    payload = feed.snapshot()
    assert payload["present"] is False
    assert payload["bpm"] is None
    assert payload["state"] == "no presence"


# -- cold start ---------------------------------------------------------


def test_the_feed_is_usable_before_the_detector_is_fitted():
    """Fitting reads the feature cache, and building that cache takes minutes on the UNO Q.
    Nothing the user looks at may wait on it: the server has to be listening and the frame
    loop has to be consuming, or on a real sensor the frames pile up on the wire.
    """
    from respiradar.live import LiveDetector

    config = RadarConfig(sweeps_per_frame=8)
    live = LiveDetector(config, _NeverFits(), evaluate_every_s=None, fit_now=False)
    assert live.fitted is False

    frames = simulated_frames(config, breaths_per_min=14.0, realtime=False)
    for frame in itertools.islice(frames, 200):
        live.process(frame)

    # It extracted happily, and refuses to produce a verdict it has no basis for.
    assert len(live.rows) == 200
    assert live.evaluate() is False
    assert _NeverFits.asked is False


def test_fitting_flips_the_flag_and_lets_verdicts_through():
    """The other half: once fitted, the detector is actually consulted."""
    from respiradar.live import LiveDetector

    config = RadarConfig(sweeps_per_frame=8)
    live = LiveDetector(config, _NeverFits(), evaluate_every_s=None, fit_now=False)
    frames = simulated_frames(config, breaths_per_min=14.0, realtime=False)
    for frame in itertools.islice(frames, 200):
        live.process(frame)

    assert _NeverFits.asked is False  # still unfitted: not consulted
    live.fit()
    assert live.fitted is True
    live.evaluate()
    assert _NeverFits.asked is True


class _NeverFits:
    """A detector with no `fit`, so `fit_on_everything` returns it untouched and no cache is
    read. `asked` records whether anything ever called `predict`."""

    name = "test/never-fits"
    asked = False

    def __init__(self):
        type(self).asked = False  # class state, so each test starts from a known place

    def predict(self, clip):
        type(self).asked = True
        return np.zeros(len(clip.t), dtype=bool)


# -- the websocket route ------------------------------------------------
#
# These exist because the route was broken for every real client while `/` and `/snapshot`
# both worked, so nothing in this file noticed. The page loaded, the panels stayed empty, and
# the only clue was "disconnected - retrying".


def _app():
    from respiradar.webscope import create_app

    feed = ScopeFeed(iter(()), RadarConfig(sweeps_per_frame=8))
    feed.worker.stop()
    return create_app(feed)


def test_the_websocket_route_takes_the_connection_not_a_query_parameter():
    """`from __future__ import annotations` makes every annotation a string, and FastAPI
    resolves those against the MODULE's globals. With `WebSocket` imported inside
    `create_app`, `'WebSocket'` was unresolvable, so FastAPI classified the argument as a
    required query parameter. Every handshake then failed validation and was closed with
    1008, which a browser reports as HTTP 403.
    """
    route = next(r for r in _app().routes if getattr(r, "path", None) == "/ws")
    assert route.dependant.websocket_param_name == "websocket"
    assert [p.name for p in route.dependant.query_params] == []


def test_the_websocket_accepts_a_connection_and_sends_a_payload():
    """End to end over ASGI, no client library: connect, and see what comes back."""
    import asyncio

    app = _app()
    scope = {
        "type": "websocket", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "scheme": "ws", "path": "/ws", "raw_path": b"/ws", "query_string": b"",
        "root_path": "", "headers": [(b"host", b"test")], "subprotocols": [],
        "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
    }

    async def drive():
        inbox = asyncio.Queue()
        await inbox.put({"type": "websocket.connect"})
        seen = []

        async def receive():
            return await inbox.get()

        async def send(message):
            seen.append(message)
            # Two payloads is enough to know it is streaming, not just accepting once.
            if len([m for m in seen if m["type"] == "websocket.send"]) >= 2:
                raise _Enough()

        try:
            await asyncio.wait_for(app(scope, receive, send), timeout=20)
        except (_Enough, asyncio.TimeoutError, BaseExceptionGroup):
            pass
        return seen

    seen = asyncio.run(drive())
    kinds = [m["type"] for m in seen]
    assert kinds, "the app sent nothing at all"
    assert kinds[0] == "websocket.accept", f"connection was rejected: {seen[0]}"
    assert "websocket.send" in kinds, f"accepted but never sent: {kinds}"


class _Enough(Exception):
    """Stop the endpoint's infinite push loop once we have seen enough."""
