#  Copyright 2017-2022 John Snow Labs
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
"""Contains utilities for pretrained annotators and pipelines."""

import sys
import threading
import time

_TRACKER = "com.johnsnowlabs.client.aws.DownloadProgressTracker"
_STYLES = ("bar", "lines", "off")
_BAR_WIDTH = 30
_POLL_SECONDS = 0.3


def printProgress(stop):
    """Prints a spinner until ``stop()`` returns True.

    Deprecated: downloads report byte-accurate progress through :class:`DownloadProgress`.
    Kept so code that imports it from earlier releases keeps working.
    """
    states = [' | ', ' / ', ' — ', ' \\ ']
    nextc = 0
    while True:
        sys.stdout.write('\r[{}]'.format(states[nextc]))
        sys.stdout.flush()
        time.sleep(2.5)
        nextc = nextc + 1 if nextc < 3 else 0
        if stop():
            sys.stdout.write('\r[{}]'.format('OK!'))
            sys.stdout.flush()
            break

    sys.stdout.write('\n')
    return


def _resolve_style(configured, stream):
    """The style to draw in: ``bar``, ``lines`` or ``off``.

    ``bar``, ``lines`` and ``off`` are honoured as configured. ``auto``, and any value this version
    does not know, decide by where the text goes: a terminal gets the one-line bar; a notebook
    cell, a CI log or a redirected file gets one line per percent, which needs no carriage-return
    support.
    """
    setting = (configured or "").strip().lower()
    if setting in _STYLES:
        return setting
    try:
        return "bar" if stream.isatty() else "lines"
    except Exception:
        return "lines"


def _percent(done, total):
    return min(100, done * 100 // total)


def _bar_text(done, total, label):
    percent = _percent(done, total)
    filled = percent * _BAR_WIDTH // 100
    bar = "=" * filled + (">" if filled < _BAR_WIDTH else "") + " " * max(
        0, _BAR_WIDTH - filled - 1)
    return "\r  [%s] %3d%%  (%.1f / %.1f MB) %s" % (bar, percent, done / 1e6, total / 1e6, label)


def _line_text(done, total, label):
    return "  downloading %s ... %3d%%  (%.1f / %.1f MB)\n" % (
        label, _percent(done, total), done / 1e6, total / 1e6)


class DownloadProgress(object):
    """Draws progress for one pretrained download by polling the JVM's byte counters.

    pyspark starts the JVM with a piped stdin, so the JVM never detects a terminal and cannot pick
    a style for the user's screen. Python can, so in a Python session Python draws and asks the JVM
    to stay quiet until the download ends.

    Degrades to silence rather than failing: an older jar without these methods, or any py4j
    error, just means no progress. It is decoration and must never break a download.
    """

    def __init__(self, spark_context, stream=None, tracker=None):
        self._sc = spark_context
        self._stream = stream if stream is not None else sys.stdout
        self._tracker = tracker
        self._style = "off"
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self._drawn = -1
        self._saw_active = False
        self._bar_open = False
        # Jars that number their downloads let this context tell its own download from counters
        # an earlier one left behind, and follow more than one download.
        self._with_generation = False
        self._entry_generation = None
        self._generation = None
        self._finished = False

    def _find_tracker(self):
        if self._tracker is not None:
            return self._tracker
        if self._sc is None:
            return None
        return getattr(self._sc._jvm, _TRACKER)

    def __enter__(self):
        try:
            tracker = self._find_tracker()
            if tracker is None:
                return self
            style = _resolve_style(tracker.configuredStyle(), self._stream)
            if style == "off":
                return self
            tracker.setRenderedByCaller(True)
        except Exception:
            return self
        self._tracker = tracker
        self._style = style
        try:
            self._entry_generation = int(tracker.snapshotWithGeneration().split("|", 1)[0])
            self._with_generation = True
        except Exception:
            self._with_generation = False  # an older jar: snapshot() only
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        """(generation, active, done, total, label), or None if the JVM cannot be reached.

        generation is None on a jar that does not number its downloads.
        """
        try:
            if self._with_generation:
                generation, rest = self._tracker.snapshotWithGeneration().split("|", 1)
                generation = int(generation)
            else:
                generation, rest = None, self._tracker.snapshot()
            active, done, total, label = rest.split("|", 3)
            return generation, active == "1", int(done), int(total), label
        except Exception:
            return None

    def _poll(self):
        while not self._stop.wait(_POLL_SECONDS):
            state = self._read()
            if state is None:
                return
            self._follow(state)

    def _follow(self, state):
        """Draws what one snapshot says about this context's downloads."""
        generation, active, done, total, label = state
        if total <= 0:
            return
        with self._lock:
            if self._with_generation:
                if generation == self._entry_generation:
                    return  # counters an earlier download left behind
                if generation != self._generation:
                    # A new download, possibly one that began and ended between two polls:
                    # close the previous one's line and count afresh.
                    self._close_line()
                    self._generation, self._drawn, self._finished = generation, -1, False
            elif active:
                self._saw_active = True
            elif not self._saw_active:
                return  # an older jar's counters cannot be told from an earlier download's
            if self._finished:
                return
            self._draw(done, total, label)
            if not active:
                # The transfer is over while the JVM still unpacks and loads the model: finish
                # the display now, so nothing printed next lands on the bar's line.
                self._close_line()
                self._finished = True

    def _close_line(self):
        with self._lock:
            if self._bar_open:
                self._stream.write("\n")
                self._stream.flush()
                self._bar_open = False

    def _draw(self, done, total, label):
        with self._lock:
            percent = _percent(done, total)
            if percent <= self._drawn:
                return
            self._drawn = percent
            if self._style == "bar":
                self._stream.write(_bar_text(done, total, label))
                self._bar_open = True
            else:
                self._stream.write(_line_text(done, total, label))
            self._stream.flush()

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is None:
            return False
        self._thread.join(timeout=2)
        # A transfer can end between two polls; one last look lands the display on 100%. Stale
        # counters from an earlier download are ignored by _follow, as they are while polling.
        if exc[0] is None:
            state = self._read()
            if state is not None:
                self._follow(state)
        self._close_line()
        try:
            self._tracker.setRenderedByCaller(False)
        except Exception:
            pass
        return False
