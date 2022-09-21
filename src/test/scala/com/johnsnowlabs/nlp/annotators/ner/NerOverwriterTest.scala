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

package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class NerOverwriterTest extends AnyFlatSpec {



  "NeC Overwriter" should "correctly should change the tag" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val testDF =
      Seq(" john Doe lives in texas.floria, ").toDF(
        "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("inplace_full")

    val document_normalizer = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("document_normalized")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document_normalized")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setMinLength(2)
      .setSplitChars(Array("-", ","))
      .setContextChars(Array("(", ")", "?", "!"))

    val normalize = new Normalizer()
      .setInputCols("token")
      .setOutputCol("token_normalized")
      .setSlangMatchCase(true)

    val embeddings = BertEmbeddings
      .pretrained("small_bert_L2_128")
      .setInputCols("sentence", "token_normalized")
      .setOutputCol("embeddings")

    val ner = NerDLModel
      .pretrained("onto_small_bert_L2_128")
      .setInputCols("sentence", "token_normalized", "embeddings")
      .setOutputCol("ner")
      .setIncludeConfidence(false)
      .setIncludeAllConfidenceScores(false)

    val converter = new NerOverwriter()
      .setInputCols("ner")
      .setOutputCol("ner2")
      .setReplaceWords(Map("B-PERSON" -> "B-PER2"))

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          document_normalizer,
          sentenceDetector,
          tokenizer,
          normalize,
          embeddings,
          ner,
          converter))

    val nermodel = pipeline.fit(testDF).transform(testDF)

    nermodel.select("ner2.result").show(1, truncate = false)

  }
}
