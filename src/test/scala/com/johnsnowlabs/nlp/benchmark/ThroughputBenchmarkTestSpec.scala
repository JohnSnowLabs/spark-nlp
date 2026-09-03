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

package com.johnsnowlabs.nlp.benchmark

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.storage.StorageLevel
import org.scalatest.flatspec.AnyFlatSpec

class ThroughputBenchmarkTestSpec extends AnyFlatSpec {

  private val spark = SparkAccessor.spark
  import spark.implicits._

  private val data = Seq(
    "Peter lives in New York. He works at Acme Corp.",
    "The quick brown fox jumps over the lazy dog.",
    "Spark NLP runs on top of Apache Spark.").toDF("text")

  private val pipelineModel = {
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentenceDetector =
      new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer)).fit(data)
  }

  "Benchmark.throughput" should "report a rate for every annotator-typed output column" taggedAs FastTest in {
    val report = Benchmark.throughput(pipelineModel, data, trials = 2)

    val byType = report.rates.map(r => r.annotatorType -> r).toMap
    assert(byType.contains(AnnotatorType.DOCUMENT))
    assert(byType.contains(AnnotatorType.TOKEN))
    assert(report.rates.forall(_.meanItemsPerSecond > 0))
    assert(report.rates.forall(_.totalItems > 0))
    assert(report.trialSeconds.length == 2)
  }

  it should "not report a rate for columns already present in the input" taggedAs FastTest in {
    val report = Benchmark.throughput(pipelineModel, data, trials = 1)
    assert(!report.rates.exists(_.outputColumn == "text"))
  }

  it should "fail fast when the input is missing the configured text column" taggedAs FastTest in {
    assertThrows[IllegalArgumentException] {
      Benchmark.throughput(pipelineModel, data, textCol = "doesNotExist")
    }
  }

  it should "cache data for the duration of the call and unpersist it again afterwards" taggedAs FastTest in {
    assert(data.storageLevel == StorageLevel.NONE, "precondition: data starts uncached")

    Benchmark.throughput(pipelineModel, data, trials = 1)

    assert(
      data.storageLevel == StorageLevel.NONE,
      "throughput should leave data it cached itself unpersisted once done")
  }

  it should "leave caller-managed caching untouched" taggedAs FastTest in {
    data.persist()
    try {
      assert(data.storageLevel != StorageLevel.NONE, "precondition: caller cached data")

      Benchmark.throughput(pipelineModel, data, trials = 1)

      assert(
        data.storageLevel != StorageLevel.NONE,
        "throughput must not unpersist a DataFrame the caller cached themselves")
    } finally data.unpersist()
  }
}
