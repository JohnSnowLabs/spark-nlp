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

class ParallelDownloadSizingTest extends AnyFlatSpec {

  private val MiB = 1024L * 1024
  private val Lengths =
    Seq(1L, 4 * MiB - 1, 4 * MiB, 8 * MiB, 8 * MiB + 1, 40 * MiB, 5L * 1024 * MiB + 3)
  private val Connections = Seq(1, 2, 3, 7, 10, 1000)

  "byteRanges" should "cover every byte exactly once, from 0 to the last" taggedAs FastTest in {
    for (length <- Lengths; connections <- Connections) {
      val ranges = AWSGateway.byteRanges(length, connections)
      val context = s"length=$length connections=$connections first=${ranges.take(2)}"
      assert(ranges.size == math.min(connections.toLong, length), context)
      assert(ranges.head._1 == 0L, context)
      assert(ranges.last._2 == length - 1, context)
      assert(ranges.forall { case (start, end) => start <= end }, context)
      ranges.zip(ranges.tail).foreach { case ((_, end), (nextStart, _)) =>
        assert(nextStart == end + 1, s"gap or overlap at $end -> $nextStart; $context")
      }
      assert(ranges.map { case (start, end) => end - start + 1 }.sum == length, context)
    }
  }

  it should "give the remainder to the last range" taggedAs FastTest in {
    assert(AWSGateway.byteRanges(10L, 3) == Seq((0L, 2L), (3L, 5L), (6L, 9L)))
  }

  it should "reject an empty object or zero connections" taggedAs FastTest in {
    assertThrows[IllegalArgumentException](AWSGateway.byteRanges(0L, 4))
    assertThrows[IllegalArgumentException](AWSGateway.byteRanges(10L, 0))
  }

  "clampDownloadThreads" should "keep download_threads between 1 and the cap" taggedAs FastTest in {
    val cap = AWSGateway.MaxParallelConnections
    assert(AWSGateway.clampDownloadThreads(-5) == 1)
    assert(AWSGateway.clampDownloadThreads(0) == 1)
    assert(AWSGateway.clampDownloadThreads(1) == 1)
    assert(AWSGateway.clampDownloadThreads(10) == 10)
    assert(AWSGateway.clampDownloadThreads(cap) == cap)
    assert(AWSGateway.clampDownloadThreads(cap + 1) == cap)
    assert(AWSGateway.clampDownloadThreads(Int.MaxValue) == cap)
  }

  "connectionsFor" should "open at least one connection and never more than contentLength / 4 MiB" taggedAs FastTest in {
    val requestedValues = Seq(0, 1, 10, AWSGateway.MaxParallelConnections + 1)
    val lengths = (Seq(-1L, 0L) ++ Lengths) :+ (8 * MiB - 1)
    for (requested <- requestedValues; length <- lengths) {
      val threads = AWSGateway.clampDownloadThreads(requested)
      val connections = AWSGateway.connectionsFor(length, threads)
      val context = s"requested=$requested length=$length -> $connections"
      assert(connections >= 1, context)
      assert(connections <= threads, context)
      assert(connections <= math.max(1L, length / AWSGateway.MinRangeBytes), context)
    }
  }

  it should "split only objects of at least two whole ranges" taggedAs FastTest in {
    assert(AWSGateway.connectionsFor(8 * MiB - 1, 10) == 1)
    assert(AWSGateway.connectionsFor(8 * MiB, 10) == 2)
    assert(AWSGateway.connectionsFor(40 * MiB, 10) == 10)
    assert(AWSGateway.connectionsFor(5L * 1024 * MiB, 1) == 1)
    assert(AWSGateway.connectionsFor(-1L, 10) == 1)
  }
}
