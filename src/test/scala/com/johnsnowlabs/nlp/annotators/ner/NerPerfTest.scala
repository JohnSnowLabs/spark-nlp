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
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class NerPerfTest extends AnyFlatSpec {

  "NerCRF Approach" should "be fast to train" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")

    val pos = PerceptronModel.pretrained().setInputCols("document", "token").setOutputCol("pos")

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setStoragePath("src/test/resources/ner-corpus/embeddings.100d.test.txt", "TEXT")
      .setDimension(100)

    val ner = new NerCrfApproach()
      .setInputCols("document", "token", "pos", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMinEpochs(1)
      .setMaxEpochs(5)
      .setC0(1250000)
      .setRandomSeed(0)
      .setVerbose(2)

    val finisher = new Finisher().setInputCols("ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, pos, embeddings, ner, finisher))

    val conll = CoNLL()
    val training_data = conll.readDataset(
      ResourceHelper.spark,
      "src/test/resources/ner-corpus/test_ner_dataset.txt")
    val nermodel = pipeline.fit(training_data)
    val nerlpmodel = new LightPipeline(nermodel)

    val res = Benchmark.time("Light annotate NerCRF") {
      nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")
    }

    println(res.mapValues(_.mkString(", ")).mkString(", "))

  }

  "NerDL Approach" should "be fast to train" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setStoragePath("src/test/resources/ner-corpus/embeddings.100d.test.txt", "TEXT")
      .setDimension(100)

    val ner = new NerDLApproach()
      .setInputCols("document", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMinEpochs(1)
      .setMaxEpochs(30)
      .setRandomSeed(0)
      .setVerbose(2)
      .setDropout(0.8f)
      .setBatchSize(18)
      .setGraphFolder("src/test/resources/graph/")

    val finisher = new Finisher().setInputCols("ner")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings, ner, finisher))

    val conll = CoNLL()
    val training_data = conll.readDataset(
      ResourceHelper.spark,
      "src/test/resources/ner-corpus/test_ner_dataset.txt")
    val nermodel = pipeline.fit(training_data)
    val nerlpmodel = new LightPipeline(nermodel)

    val res = Benchmark.time("Light annotate NerDL") {
      nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")
    }

    println(res.mapValues(_.mkString(", ")).mkString(", "))

    nermodel.stages(3).asInstanceOf[NerDLModel].write.overwrite().save("./tmp_nerdl")

  }

  "NerDL Model" should "label correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val ner =
      NerDLModel.pretrained().setInputCols("sentence", "token", "embeddings").setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("nerconverter")

    val finisher = new Finisher().setInputCols("token", "sentence", "nerconverter", "ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizer, embeddings, ner, converter, finisher))

    val conll = CoNLL()
    val training_data = conll.readDataset(
      ResourceHelper.spark,
      "src/test/resources/ner-corpus/test_ner_dataset.txt")
    val nermodel = pipeline.fit(training_data)
    val nerlpmodel = new LightPipeline(nermodel)

    val res1 = Benchmark.time("Light annotate NerDL") {
      nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")
    }
    val res2 = Benchmark.time("Light annotate NerDL") {
      nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")
    }
    val res3 = Benchmark.time("Light annotate NerDL") {
      nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70 yearold")
    }
    val res4 = Benchmark.time("Light annotate NerDL") {
      nerlpmodel.fullAnnotate("Ms.")
    }

    println(res1.mapValues(_.mkString(", ")).mkString(", "))
    println(res2.mapValues(_.mkString(", ")).mkString(", "))
    println(res3.mapValues(_.mkString(", ")).mkString(", "))
    println(res4.mapValues(_.mkString(", ")).mkString(", "))

  }

  "NerCRF Model" should "label correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")

    val pos = PerceptronModel.pretrained().setInputCols("document", "token").setOutputCol("pos")

    val word_embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    // document, token, pos, word_embeddings
    val ner = NerCrfModel
      .pretrained()
      .setInputCols("sentence", "token", "pos", "word_embeddings")
      .setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("nerconverter")

    val finisher = new Finisher().setInputCols("token", "sentence", "nerconverter", "ner")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        pos,
        word_embeddings,
        ner,
        converter,
        finisher))

    val conll = CoNLL()
    val training_data = conll.readDataset(
      ResourceHelper.spark,
      "src/test/resources/ner-corpus/test_ner_dataset.txt")
    val nermodel = pipeline.fit(training_data)
    val nerlpmodel = new LightPipeline(nermodel)

    val res1 = Benchmark.time("Light annotate NerCrf") {
      nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")
    }
    val res2 = Benchmark.time("Light annotate NerCrf") {
      nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")
    }
    val res3 = Benchmark.time("Light annotate NerCrf") {
      nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70yearold")
    }
    val res4 = Benchmark.time("Light annotate NerCrf") {
      nerlpmodel.fullAnnotate("Ms.")
    }

    println(res1.mapValues(_.mkString(", ")).mkString(", "))
    println(res2.mapValues(_.mkString(", ")).mkString(", "))
    println(res3.mapValues(_.mkString(", ")).mkString(", "))
    println(res4.mapValues(_.mkString(", ")).mkString(", "))

  }

}
