/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.tags

import org.scalatest.Tag

object FastTest extends Tag("com.johnsnowlabs.tags.FastTest")
object SlowTest extends Tag("com.johnsnowlabs.tags.SlowTest")

// Opt-in pilot dimensions; existing tests need not carry these tags.
object ONNX extends Tag("com.johnsnowlabs.tags.ONNX")
object TensorFlow extends Tag("com.johnsnowlabs.tags.TensorFlow")
object Embeddings extends Tag("com.johnsnowlabs.tags.Embeddings")
object Text extends Tag("com.johnsnowlabs.tags.Text")
object ResourceIntensiveTest extends Tag("com.johnsnowlabs.tags.ResourceIntensiveTest")

object TestTaxonomy {
  val tags: Map[String, Tag] = Seq(
    FastTest,
    SlowTest,
    ONNX,
    TensorFlow,
    Embeddings,
    Text,
    ResourceIntensiveTest).map(tag => tag.name.split('.').last -> tag).toMap

  /** Accept Scala tag names and the Python marker vocabulary used by teammates. */
  val names: Map[String, Tag] = tags ++ Map(
    "fast" -> FastTest,
    "slow" -> SlowTest,
    "onnx" -> ONNX,
    "tensorflow" -> TensorFlow,
    "embeddings" -> Embeddings,
    "text" -> Text,
    "resource_intensive" -> ResourceIntensiveTest)
}
