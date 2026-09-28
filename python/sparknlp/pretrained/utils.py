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
_STYLE = "com.johnsnowlabs.client.aws.ProgressStyle"
_BAR_WIDTH = 30
_POLL_SECONDS = 0.3


def _in_notebook():
    """Whether this is a notebook kernel rather than a terminal.

    Only there does Python need to render: a Spark driver launched through py4j writes to a
    stdout the notebook front end does not surface, so the JVM's own bar is invisible however
    it is formatted. In a terminal the JVM draws it and this stays quiet, so the two never
    write to the same stream.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        # Test for a kernel rather than for a class name. Jupyter's shell is
        # ZMQInteractiveShell, but Colab subclasses it as google.colab._shell.Shell, and a
        # name comparison silently excludes exactly the environment this exists for. A
        # terminal IPython shell has no kernel attribute.
        return shell is not None and hasattr(shell, "kernel")
    except Exception:
        return False


def _render(done, total, label):
    percent = min(100, done * 100 // total)
    filled = percent * _BAR_WIDTH // 100
    bar = "=" * filled + (">" if filled < _BAR_WIDTH else "") + " " * max(
        0, _BAR_WIDTH - filled - 1)
    sys.stdout.write(
        "\r  [%s] %3d%%  (%.1f / %.1f MB) %s" % (bar, percent, done / 1e6, total / 1e6, label))
    sys.stdout.flush()


class DownloadProgress(object):
    """Renders download progress in a notebook by polling the JVM's byte counters.

    Python cannot observe the transfer directly -- it is blocked in a py4j call for the whole
    of it -- so the JVM publishes counters and this polls them from a background thread.

    Degrades to silence rather than failing: an older jar without the counters, a py4j hiccup,
    or a terminal session all just mean no bar. It is decoration, and must never be able to
    break a download.
    """

    def __init__(self, spark_context):
        self._sc = spark_context
        self._stop = threading.Event()
        self._thread = None
        self._tracker = None
        self._drawn = -1

    def __enter__(self):
        if not _in_notebook() or self._sc is None:
            return self
        try:
            jvm = self._sc._jvm
            tracker = getattr(jvm, _TRACKER)
            if getattr(jvm, _STYLE).resolved() == "off":
                return self
        except Exception:
            return self
        self._tracker = tracker
        self._thread = threading.Thread(target=self._poll, args=(tracker,), daemon=True)
        self._thread.start()
        return self

    @staticmethod
    def _read(tracker):
        """(active, done, total, label), or None if the JVM cannot be reached."""
        try:
            active, done, total, label = tracker.snapshot().split("|", 3)
        except Exception:
            return None
        return active == "1", int(done), int(total), label

    def _poll(self, tracker):
        while not self._stop.wait(_POLL_SECONDS):
            state = self._read(tracker)
            if state is None:
                return
            active, done, total, label = state
            if active and total > 0:
                self._draw(done, total, label)

    def _draw(self, done, total, label):
        percent = min(100, done * 100 // total)
        if percent <= self._drawn:
            return
        self._drawn = percent
        _render(done, total, label)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is None:
            return False
        self._thread.join(timeout=2)
        # One last read: the transfer finishes between polls, so without this the bar stops
        # wherever the final poll happened to land rather than at 100%.
        if exc[0] is None and self._tracker is not None:
            state = self._read(self._tracker)
            if state is not None and state[2] > 0:
                self._draw(state[1], state[2], state[3])
        sys.stdout.write("\n")
        sys.stdout.flush()
        return False
