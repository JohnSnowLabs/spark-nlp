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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, SparkSession}
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkSessionTest extends BeforeAndAfterAll { this: Suite =>

  val spark: SparkSession = SparkAccessor.spark
  val tokenizerPipeline = new Pipeline()
  val tokenizerWithSentencePipeline = new Pipeline()
  val documentAssembler = new DocumentAssembler()
  val sentenceDetector = new SentenceDetector()
  val tokenizer = new Tokenizer()
  val tokenizerWithSentence = new Tokenizer()
  val emptyDataSet: Dataset[_] = PipelineModels.dummyDataset
  val pipeline = new Pipeline()

  override def beforeAll(): Unit = {
    super.beforeAll()

    documentAssembler.setInputCol("text").setOutputCol("document")
    tokenizer.setInputCols("document").setOutputCol("token")
    tokenizerPipeline.setStages(Array(documentAssembler, tokenizer))

    sentenceDetector.setInputCols("document").setOutputCol("sentence")
    tokenizerWithSentence.setInputCols("sentence").setOutputCol("token")
    tokenizerWithSentencePipeline.setStages(
      Array(documentAssembler, sentenceDetector, tokenizerWithSentence))
  }

}
