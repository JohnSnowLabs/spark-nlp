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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkSessionTest extends BeforeAndAfterAll { this: Suite =>

  val spark: SparkSession = SparkAccessor.spark
  val tokenizerPipeline = new Pipeline()
  val tokenizerWithSentencePipeline = new Pipeline()
  val documentAssembler = new DocumentAssembler()
  val tokenizer = new Tokenizer()
  val emptyDataSet: Dataset[_] = PipelineModels.dummyDataset

  override def beforeAll(): Unit = {
    super.beforeAll()

    documentAssembler.setInputCol("text").setOutputCol("document")
    tokenizer.setInputCols("document").setOutputCol("token")
    tokenizerPipeline.setStages(Array(documentAssembler, tokenizer))

    val sentenceDetector = new SentenceDetector()
    sentenceDetector.setInputCols("document").setOutputCol("sentence")
    val tokenizerWithSentence = new Tokenizer()
    tokenizerWithSentence.setInputCols("sentence").setOutputCol("token")
    tokenizerWithSentencePipeline.setStages(Array(documentAssembler, sentenceDetector, tokenizerWithSentence))
  }

}
