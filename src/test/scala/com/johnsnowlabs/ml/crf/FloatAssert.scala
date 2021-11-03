/*
 * Copyright 2017-2019 John Snow Labs
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

package com.johnsnowlabs.ml.crf

object FloatAssert {

  def seqEquals(a: Seq[Float], b: Seq[Float], eps: Float = 1e-7f): Unit = {
    assert(a.size == b.size, s"$a is not equal $b")

    for (i <- 0 until a.size)
      assert(Math.abs(a(i) - b(i)) <= eps, s"$a does not equal $b\nExpected\t:$b\nActual\t\t:$a\n")
  }

  def equals(a: Float, b: Float, eps: Float = 1e-7f): Unit = {
    assert(Math.abs(a - b) <= eps, s"$a does not equal $b\nExpected\t:$b\nActual\t\t:$a\n")
  }
}
