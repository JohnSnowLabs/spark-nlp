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

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.training.CoNLLU
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Paths, Files}

class LemmatizerTestSpec extends AnyFlatSpec with LemmatizerBehaviors {

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  require(Some(SparkAccessor).isDefined)

  val lemmatizer = new Lemmatizer
  "a lemmatizer" should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    assert(lemmatizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullLemmatizerPipeline(
    latinBodyData)

  "A lemmatizer" should "be readable and writable" taggedAs Tag("LinuxOnly") in {
    val lemmatizer = new Lemmatizer()
      .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")
    val path = "./test-output-tmp/lemmatizer"
    try {
      lemmatizer.write.overwrite.save(path)
      val lemmatizerRead = Lemmatizer.read.load(path)
      assert(lemmatizer.getDictionary.path == lemmatizerRead.getDictionary.path)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

  "A lemmatizer" should "work under a pipeline framework" taggedAs FastTest in {

    val data = ContentProvider.parquetData.limit(1000)

    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

    val finisher = new Finisher()
      .setInputCols("lemma")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer, finisher))

    val recursivePipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer, finisher))

    val model = pipeline.fit(data)
    model.transform(data).show(1)

    val PIPE_PATH = "./tmp_pipeline"

    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(data).show(1)

    val recursiveModel = recursivePipeline.fit(data)
    recursiveModel.transform(data).show(1)

    recursiveModel.write.overwrite().save(PIPE_PATH)
    val loadedRecPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedRecPipeline.transform(data).show(1)

    succeed
  }

  import SparkAccessor.spark.implicits._

  it should "lemmatize text from a spark dataset" taggedAs FastTest in {
    val testDataSet = Seq("So what happened?", "That too was stopped.").toDS.toDF("text")
    val expectedLemmas = Array(
      Seq(
        Annotation(TOKEN, 0, 1, "So", Map("sentence" -> "0")),
        Annotation(TOKEN, 3, 6, "what", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 15, "happen", Map("sentence" -> "0")),
        Annotation(TOKEN, 16, 16, "?", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 3, "That", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 7, "too", Map("sentence" -> "0")),
        Annotation(TOKEN, 9, 11, "be", Map("sentence" -> "0")),
        Annotation(TOKEN, 13, 19, "stop", Map("sentence" -> "0")),
        Annotation(TOKEN, 20, 20, ".", Map("sentence" -> "0"))))
    val conlluFile = "src/test/resources/conllu/en.test.lemma.conllu"
    val trainDataSet = CoNLLU(formCol = "form_training", lemmaCol = "lemma_training")
      .readDataset(ResourceHelper.spark, conlluFile)
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setFormCol("form_training")
      .setLemmaCol("lemma_training")
      .setOutputCol("lemma")
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val model = pipeline.fit(trainDataSet)
    val lemmaDataSet = model.transform(testDataSet)

    assertLemmas(lemmaDataSet, expectedLemmas)
  }

  it should "raise error when form column is not present" taggedAs FastTest in {
    val testDataSet = Seq(("text column", "lemma column")).toDS.toDF("text", "lemma")
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val caught = intercept[IllegalArgumentException] {
      pipeline.fit(testDataSet)
    }

    assert(
      caught.getMessage == "form column required. Verify that training dataset was loaded with CoNLLU component")
  }

  it should "raise error when lemma column is not present" taggedAs FastTest in {
    val testDataSet = Seq("text data").toDS.toDF("text")
    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("form")
    val tokenizerPipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer))
    val tokenizerDataSet = tokenizerPipeline.fit(testDataSet).transform(testDataSet)
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("form"))
      .setOutputCol("lemma")
    val lemmatizerPipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, lemmatizer))

    val caught = intercept[IllegalArgumentException] {
      lemmatizerPipeline.fit(tokenizerDataSet)
    }

    assert(
      caught.getMessage == "lemma column required. Verify that training dataset was loaded with CoNLLU component")
  }

  it should "raise error when lemma or form does not have token annotator type" taggedAs FastTest in {
    val testDataSet =
      Seq(("text column", "form column", "lemma column")).toDS.toDF("text", "form", "lemma")
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val caught = intercept[IllegalArgumentException] {
      pipeline.fit(testDataSet)
    }

    assert(caught.getMessage == "form is not a token annotator type")
  }

  it should "serialize a lemmatizer model" taggedAs FastTest in {
    val conlluFile = "src/test/resources/conllu/en.test.lemma.conllu"
    val trainDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    val lemmatizerModel = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .fit(trainDataSet)

    lemmatizerModel.write.overwrite().save("./tmp_lemmatizer")

    assertResult(true) {
      Files.exists(Paths.get("./tmp_lemmatizer"))
    }
  }

  it should "deserialize a lemmatizer model" taggedAs FastTest in {
    val testDataSet = Seq("So what happened?").toDS.toDF("text")
    val expectedLemmas = Array(
      Seq(
        Annotation(TOKEN, 0, 1, "So", Map("sentence" -> "0")),
        Annotation(TOKEN, 3, 6, "what", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 15, "happen", Map("sentence" -> "0")),
        Annotation(TOKEN, 16, 16, "?", Map("sentence" -> "0"))))
    val conlluFile = "src/test/resources/conllu/en.test.lemma.conllu"
    val trainDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    val lemmatizer = LemmatizerModel.load("./tmp_lemmatizer")
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val model = pipeline.fit(trainDataSet)
    val lemmaDataSet = model.transform(testDataSet)

    assertLemmas(lemmaDataSet, expectedLemmas)
  }

  private def assertLemmas(
      lemmaDataSet: Dataset[_],
      expectedLemmas: Array[Seq[Annotation]]): Unit = {
    val actualLemmas = AssertAnnotations.getActualResult(lemmaDataSet, "lemma")
    assert(actualLemmas.length == expectedLemmas.length)
    AssertAnnotations.assertFields(expectedLemmas, actualLemmas)
  }

  it should "download pretrained model" taggedAs FastTest in {
    val testDataSet = Seq("So what happened?").toDS.toDF("text")
    val lemmatizer = LemmatizerModel
      .pretrained()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val model = pipeline.fit(testDataSet)
    model.transform(testDataSet).show()
  }
}
