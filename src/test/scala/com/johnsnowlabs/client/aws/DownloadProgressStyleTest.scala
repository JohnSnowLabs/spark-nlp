/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.client.aws

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{ByteArrayOutputStream, PrintStream}

class DownloadProgressStyleTest extends AnyFlatSpec {

  private def capture(style: String, total: Long)(steps: DownloadProgress => Unit): String = {
    val out = new ByteArrayOutputStream()
    steps(new DownloadProgress(total, "m.zip", new PrintStream(out, true, "UTF-8"), style))
    out.toString("UTF-8")
  }

  "ProgressStyle.normalise" should "keep bar, lines and off and map anything else to auto" taggedAs FastTest in {
    assert(ProgressStyle.normalise("bar") == "bar")
    assert(ProgressStyle.normalise(" LINES ") == "lines")
    assert(ProgressStyle.normalise("Off") == "off")
    Seq("auto", "", null, "fancy").foreach { setting =>
      assert(ProgressStyle.normalise(setting) == "auto", s"setting=$setting")
    }
  }

  "ProgressStyle.fromConfig" should "honour explicit styles and resolve auto by the terminal" taggedAs FastTest in {
    for (terminal <- Seq(true, false); style <- Seq("bar", "lines", "off"))
      assert(ProgressStyle.fromConfig(style, terminal) == style)
    assert(ProgressStyle.fromConfig("auto", terminal = true) == "bar")
    assert(ProgressStyle.fromConfig("auto", terminal = false) == "lines")
    assert(ProgressStyle.fromConfig("fancy", terminal = true) == "bar")
    assert(ProgressStyle.fromConfig(null, terminal = false) == "lines")
  }

  "DownloadProgress" should "draw nothing while the caller renders the download" taggedAs FastTest in {
    val text = capture(ProgressStyle.Lines, 100L) { progress =>
      DownloadProgressTracker.setRenderedByCaller(true)
      try progress.advance(50L)
      finally DownloadProgressTracker.setRenderedByCaller(false)
    }
    assert(text.isEmpty)
    assert(capture(ProgressStyle.Lines, 100L)(_.advance(100L)).contains("100%"))
  }

  it should "print at most one line per percent in lines style" taggedAs FastTest in {
    val text = capture(ProgressStyle.Lines, 1000L)(p => (1 to 1000).foreach(_ => p.advance(1L)))
    val lines = text.split("\n").filter(_.nonEmpty)
    assert(lines.length <= 101)
    assert(!text.contains("\r"))
    assert(lines.last.contains("100%"))
  }

  it should "repaint one line in bar style and end it at 100%" taggedAs FastTest in {
    val text = capture(ProgressStyle.Bar, 1000L)(p => (1 to 1000).foreach(_ => p.advance(1L)))
    assert(text.contains("\r"))
    assert(text.count(_ == '\n') == 1)
    assert(text.endsWith("\n"))
  }

  "DownloadProgressTracker.configuredStyle" should "report the setting itself, not the style resolved from it" taggedAs FastTest in {
    // Nothing configures download_progress in the test JVM, so the setting is its default.
    // Python resolves auto against its own stdout; handing it "lines" here would decide for it.
    assert(DownloadProgressTracker.configuredStyle == "auto")
  }

  "DownloadProgressTracker.setRenderedByCaller" should "keep the JVM quiet until every caller that set it has cleared it" taggedAs FastTest in {
    DownloadProgressTracker.setRenderedByCaller(true)
    DownloadProgressTracker.setRenderedByCaller(true)
    try {
      DownloadProgressTracker.setRenderedByCaller(false)
      assert(
        DownloadProgressTracker.isRenderedByCaller,
        "the other notebook cell is still drawing")
    } finally DownloadProgressTracker.setRenderedByCaller(false)
    assert(!DownloadProgressTracker.isRenderedByCaller)
  }

  "DownloadProgressTracker.snapshotWithGeneration" should "start a new generation with every download" taggedAs FastTest in {
    val before = DownloadProgressTracker.snapshotWithGeneration.split('|')(0).toLong
    DownloadProgressTracker.begin("a.zip", 10L)
    DownloadProgressTracker.end()
    DownloadProgressTracker.begin("b.zip", 20L)
    try {
      assert(
        DownloadProgressTracker.snapshotWithGeneration.split("\\|", 5).toSeq ==
          Seq((before + 2).toString, "1", "0", "20", "b.zip"))
    } finally DownloadProgressTracker.end()
  }
}
