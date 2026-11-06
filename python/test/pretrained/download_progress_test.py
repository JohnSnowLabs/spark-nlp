#  Copyright 2017-2026 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import io
import re
import threading
import time
import unittest

import pytest

from sparknlp.pretrained import utils
from sparknlp.pretrained.utils import DownloadProgress, _resolve_style


class _Stream(io.StringIO):
    """A stdout stand-in that says whether it is a terminal."""

    def __init__(self, tty):
        super(_Stream, self).__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


class _FakeTracker(object):
    """Stands in for the JVM's DownloadProgressTracker: replays snapshots and records calls."""

    def __init__(self, style, snapshots):
        self.style = style
        self._snapshots = list(snapshots)
        self.rendered_by_caller = []
        self.replayed = threading.Event()

    def configuredStyle(self):
        return self.style

    def setRenderedByCaller(self, value):
        self.rendered_by_caller.append(value)

    def snapshot(self):
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        self.replayed.set()
        return self._snapshots[0]


def _downloading(*percents):
    return ["1|%d|1000|m.zip" % (p * 10) for p in percents]


def _render(style, snapshots, tty):
    tracker = _FakeTracker(style, snapshots)
    stream = _Stream(tty)
    with DownloadProgress(None, stream=stream, tracker=tracker):
        if style != "off":
            assert tracker.replayed.wait(5), "the poller never drained the snapshots"
    return stream.getvalue(), tracker


class _FastPolling(unittest.TestCase):
    def setUp(self):
        self._poll = utils._POLL_SECONDS
        utils._POLL_SECONDS = 0.01

    def tearDown(self):
        utils._POLL_SECONDS = self._poll


@pytest.mark.fast
class ResolveStyleTestSpec(unittest.TestCase):
    def runTest(self):
        for style in ("bar", "lines", "off"):
            self.assertEqual(_resolve_style(style, _Stream(tty=True)), style)
            self.assertEqual(_resolve_style(style, _Stream(tty=False)), style)
        self.assertEqual(_resolve_style(" LINES ", _Stream(tty=True)), "lines")
        for style in ("auto", "", None, "fancy"):
            self.assertEqual(_resolve_style(style, _Stream(tty=True)), "bar")
            self.assertEqual(_resolve_style(style, _Stream(tty=False)), "lines")


@pytest.mark.fast
class LinesStyleTestSpec(_FastPolling):
    def runTest(self):
        output, tracker = _render("lines", _downloading(0, 10, 50, 100), tty=True)
        self.assertNotIn("\r", output)
        self.assertEqual(re.findall(r"(\d+)%", output), ["0", "10", "50", "100"])
        self.assertTrue(all(line.startswith("  downloading m.zip") for line in output.splitlines()))
        self.assertEqual(tracker.rendered_by_caller, [True, False])


@pytest.mark.fast
class BarStyleTestSpec(_FastPolling):
    def runTest(self):
        output, tracker = _render("bar", _downloading(0, 50, 100), tty=False)
        self.assertIn("\r", output)
        self.assertEqual(output.count("\n"), 1)
        self.assertTrue(output.endswith("\n"))
        self.assertIn("100%", output)
        self.assertEqual(tracker.rendered_by_caller, [True, False])


@pytest.mark.fast
class AutoStyleTestSpec(_FastPolling):
    def runTest(self):
        terminal, _ = _render("auto", _downloading(0, 100), tty=True)
        notebook, _ = _render("auto", _downloading(0, 100), tty=False)
        self.assertIn("\r", terminal)
        self.assertNotIn("\r", notebook)
        self.assertEqual(re.findall(r"(\d+)%", notebook), ["0", "100"])


@pytest.mark.fast
class OffStyleTestSpec(_FastPolling):
    def runTest(self):
        output, tracker = _render("off", _downloading(0, 100), tty=True)
        self.assertEqual(output, "")
        self.assertEqual(tracker.rendered_by_caller, [])


@pytest.mark.fast
class OlderJarTestSpec(_FastPolling):
    def runTest(self):
        stream = _Stream(tty=True)
        with DownloadProgress(None, stream=stream, tracker=object()):
            pass
        self.assertEqual(stream.getvalue(), "")


@pytest.mark.fast
class FailedDownloadTestSpec(_FastPolling):
    def runTest(self):
        tracker = _FakeTracker("lines", _downloading(0, 40))
        stream = _Stream(tty=False)
        with self.assertRaises(ValueError):
            with DownloadProgress(None, stream=stream, tracker=tracker):
                tracker.replayed.wait(5)
                raise ValueError("download failed")
        self.assertEqual(tracker.rendered_by_caller, [True, False])


@pytest.mark.fast
class StaleCountersTestSpec(_FastPolling):
    def runTest(self):
        # The model was already cached: the JVM still holds the previous download's counters.
        output, _ = _render("lines", ["0|1000|1000|previous.zip"], tty=False)
        self.assertEqual(output, "")


def _output_once_transfer_ends(style, tty, done):
    """Output written while the context is still open: the transfer has ended (the tracker went
    inactive at 100%) but the JVM is still unpacking and loading the model."""
    tracker = _FakeTracker(style, _downloading(0, 50) + ["0|1000|1000|m.zip"])
    stream = _Stream(tty)
    with DownloadProgress(None, stream=stream, tracker=tracker):
        tracker.replayed.wait(5)
        deadline = time.time() + 5
        while not done(stream.getvalue()) and time.time() < deadline:
            time.sleep(0.01)
        during = stream.getvalue()
    return during, stream.getvalue()


@pytest.mark.fast
class BarClosedWhenTransferEndsTestSpec(_FastPolling):
    def runTest(self):
        during, after = _output_once_transfer_ends("bar", True, lambda out: out.endswith("\n"))
        self.assertTrue(during.endswith("\n"), "the bar line must close as soon as the transfer ends")
        self.assertIn("100%", during)
        self.assertEqual(after.count("\n"), 1, "no second 100% line after the model loads")


@pytest.mark.fast
class LastLineWhenTransferEndsTestSpec(_FastPolling):
    def runTest(self):
        during, after = _output_once_transfer_ends("lines", False, lambda out: "100%" in out)
        self.assertIn("100%", during, "a notebook must see 100% when the transfer ends, not after the load")
        self.assertEqual(re.findall(r"(\d+)%", after), ["0", "50", "100"])


class _GenerationTracker(_FakeTracker):
    """A tracker from a jar that numbers its downloads: snapshots read
    ``generation|active|done|total|label``."""

    def snapshotWithGeneration(self):
        return _FakeTracker.snapshot(self)

    def snapshot(self):
        return self.snapshotWithGeneration().split("|", 1)[1]


def _render_generations(style, snapshots, tty=False):
    tracker = _GenerationTracker(style, snapshots)
    stream = _Stream(tty)
    with DownloadProgress(None, stream=stream, tracker=tracker):
        assert tracker.replayed.wait(5), "the poller never drained the snapshots"
    return stream.getvalue()


def _drawn(output):
    return re.findall(r"downloading (\S+) \.\.\.\s+(\d+)%", output)


@pytest.mark.fast
class DownloadBetweenTwoPollsTestSpec(_FastPolling):
    def runTest(self):
        # The download started and finished between two polls; only its final state is seen.
        output = _render_generations("lines", ["5|0|1000|1000|old.zip", "6|0|2000|2000|new.zip"])
        self.assertEqual(_drawn(output), [("new.zip", "100")])


@pytest.mark.fast
class SecondDownloadInOneCallTestSpec(_FastPolling):
    def runTest(self):
        output = _render_generations("lines", [
            "5|0|1000|1000|old.zip",
            "6|1|0|1000|a.zip", "6|1|1000|1000|a.zip", "6|0|1000|1000|a.zip",
            "7|1|0|1000|b.zip", "7|1|1000|1000|b.zip", "7|0|1000|1000|b.zip"])
        self.assertEqual(
            _drawn(output), [("a.zip", "0"), ("a.zip", "100"), ("b.zip", "0"), ("b.zip", "100")])


@pytest.mark.fast
class StaleCountersWithGenerationTestSpec(_FastPolling):
    def runTest(self):
        self.assertEqual(_render_generations("lines", ["5|0|1000|1000|old.zip"]), "")


@pytest.mark.fast
class DownloadModelDirectlyShowsProgressTestSpec(unittest.TestCase):
    def runTest(self):
        from sparknlp.pretrained import resource_downloader as downloader
        calls = []

        class _Progress(object):
            def __init__(self, spark_context):
                pass

            def __enter__(self):
                calls.append("progress on")
                return self

            def __exit__(self, *exc):
                calls.append("progress off")
                return False

        class _Download(object):
            def __init__(self, *args):
                pass

            def apply(self):
                calls.append("download")

        saved = downloader.DownloadProgress, downloader._internal._DownloadModelDirectly
        downloader.DownloadProgress, downloader._internal._DownloadModelDirectly = _Progress, _Download
        try:
            downloader.ResourceDownloader.downloadModelDirectly("public/models/m.zip")
        finally:
            downloader.DownloadProgress, downloader._internal._DownloadModelDirectly = saved
        self.assertEqual(calls, ["progress on", "download", "progress off"])


@pytest.mark.fast
class PrintProgressStillImportableTestSpec(unittest.TestCase):
    def runTest(self):
        from sparknlp.pretrained import printProgress
        self.assertTrue(callable(printProgress))
