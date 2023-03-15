/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper
import scala.math.*
import com.johnsnowlabs.ml.ai.util.Generation.Logit.Logit
abstract class LogitWarper extends Logit {

  protected def scanLeft[a, b](xs: Iterable[a])(s: b)(f: (b, a) => b): Seq[b] =
    xs.foldLeft(List(s))((acc, x) => f(acc.head, x) :: acc).reverse

  protected def softmax(values: Array[Float]): Array[Float] = {
    val expElem = values.map(exp(_))
    val total = expElem.sum
    expElem.map(_ / total).map(_.toFloat)
  }

  protected def scatterValuesOnBatchIndices(
      values: List[Boolean],
      batchIndices: Array[Int]): List[Boolean] = {
    // scatter values to pair indices
    val (_, initArray) = batchIndices.zip(values).sorted.unzip
    initArray.toList
  }
}
