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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class Doc2ChunkTestSpec extends AnyFlatSpec {

  "a chunk assembler" should "correctly chunk ranges" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    val sampleDataset = Seq[(String, String)](
      ("Hello world, this is a sentence out of nowhere", "a sentence out"),
      ("Hey there, there is no chunk here", ""),
      ("Woah here, don't go so fast", "this is not there")).toDF("sentence", "target")

    val answer = Array(
      Seq[Annotation](
        Annotation(
          AnnotatorType.CHUNK,
          21,
          34,
          "a sentence out",
          Map("sentence" -> "0", "chunk" -> "0"))),
      Seq.empty[Annotation],
      Seq.empty[Annotation])

    val documentAssembler =
      new DocumentAssembler().setInputCol("sentence").setOutputCol("document")

    val chunkAssembler =
      new Doc2Chunk().setInputCols("document").setChunkCol("target").setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler))

    val results = pipeline
      .fit(Seq.empty[(String, String)].toDF("sentence", "target"))
      .transform(sampleDataset)
      .select("chunk")
      .as[Seq[Annotation]]
      .collect()

    for ((a, b) <- results.zip(answer)) {
      assert(a == b)
    }

  }

  "a chunk assembler" should "correctly chunk array ranges" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    val sampleDataset = Seq[(String, Seq[String])](
      ("Hello world, this is a sentence out of nowhere", Seq("world", "out of nowhere")),
      ("Hey there, there is no chunk here", Seq.empty[String]),
      ("Woah here, don't go so fast", Seq[String]("this is not there", "so fast")))
      .toDF("sentence", "target")

    val answer = Array(
      Seq[Annotation](
        Annotation(AnnotatorType.CHUNK, 6, 10, "world", Map("sentence" -> "0", "chunk" -> "0")),
        Annotation(
          AnnotatorType.CHUNK,
          32,
          45,
          "out of nowhere",
          Map("sentence" -> "0", "chunk" -> "1"))),
      Seq.empty[Annotation],
      Seq[Annotation](
        Annotation(
          AnnotatorType.CHUNK,
          20,
          26,
          "so fast",
          Map("sentence" -> "0", "chunk" -> "1"))))

    val documentAssembler =
      new DocumentAssembler().setInputCol("sentence").setOutputCol("document")

    val chunkAssembler = new Doc2Chunk()
      .setIsArray(true)
      .setInputCols("document")
      .setChunkCol("target")
      .setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler))

    val results = pipeline
      .fit(Seq.empty[(String, Seq[String])].toDF("sentence", "target"))
      .transform(sampleDataset)
      .select("chunk")
      .as[Seq[Annotation]]
      .collect()

    for ((a, b) <- results.zip(answer)) {
      assert(a == b)
    }

  }

}
