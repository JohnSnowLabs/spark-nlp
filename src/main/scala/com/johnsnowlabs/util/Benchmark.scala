/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.util

object Benchmark {

  private var print = true

  def setPrint(v: Boolean): Unit = print = v

  def getPrint: Boolean = print

  def time[R](description: String, forcePrint: Boolean = false)(block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    if (print || forcePrint) println(description + ": " + ((t1 - t0) / 1000000000.0) + "sec")
    result
  }

  def measure(iterations: Integer = 3, forcePrint: Boolean = false, description: String = "Took")(f: => Any): Double = {
    val time = (0 until iterations).map { _ =>
      val t0 = System.nanoTime()
      f
      System.nanoTime() - t0
    }.sum.toDouble / iterations

    if (print || forcePrint) println(s"$description (Avg for $iterations iterations): ${time / 1000000000} sec")

    time / 1000000000
  }

  def measure(f: => Any): Double = measure()(f)

  def measure(d: String)(f: => Any): Double = measure(description = d)(f)
}