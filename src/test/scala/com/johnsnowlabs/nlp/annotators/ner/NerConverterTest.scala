/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import org.scalatest._

class NerConverterTest extends FlatSpec {

  "NerConverter" should "correctly use any TOKEN type input" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/ner-corpus/test_ner_dataset.txt")

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalize = new Normalizer()
      .setInputCols("token")
      .setOutputCol("cleaned")
      .setLowercase(true)

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("document", "cleaned")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val ner = NerDLModel.pretrained()
      .setInputCols("sentence", "cleaned", "embeddings")
      .setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "cleaned", "ner")
      .setOutputCol("entities")
      .setPreservePosition(false)

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalize,
        embeddings,
        ner,
        converter
      ))

    val nermodel = recursivePipeline.fit(training_data).transform(training_data)

    nermodel.select("token.result").show(1, false)
    nermodel.select("cleaned.result").show(1, false)
    nermodel.select("embeddings.result").show(1, false)
    nermodel.select("entities.result").show(1, false)
    nermodel.select("entities").show(1, false)

  }

  "NeConverter" should "correctly work in a pipeline with per-processing" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val testDF = Seq(
      "word1 word2 word3 word4.word5 john,doe........... john.doe texas.floria, "
    ).toDF("text")

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

    val embeddings = BertEmbeddings.pretrained("small_bert_L2_128")
      .setInputCols("sentence", "token_normalized")
      .setOutputCol("embeddings")

    val ner = NerDLModel.pretrained("onto_small_bert_L2_128")
      .setInputCols("sentence", "token_normalized", "embeddings")
      .setOutputCol("ner")
      .setIncludeConfidence(true)

    val converter = new NerConverter()
      .setInputCols("sentence", "token_normalized", "ner")
      .setOutputCol("entities")
      .setPreservePosition(false)

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        document_normalizer,
        sentenceDetector,
        tokenizer,
        normalize,
        embeddings,
        ner,
        converter
      ))

    val nermodel = recursivePipeline.fit(testDF).transform(testDF)

    nermodel.select("document.result").show(1, false)
    nermodel.select("document_normalized.result").show(1, false)
    nermodel.select("sentence.result").show(1, false)
    nermodel.select("token.result").show(1, false)
    nermodel.select("token_normalized.result").show(1, false)
    nermodel.select("embeddings.result").show(1, false)
    nermodel.select("entities.result").show(1, false)
    nermodel.select("entities").show(1, false)

  }
}

