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

package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.{
  DependencyParserApproach,
  DependencyParserModel
}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

class TypedDependencyModelTestSpec extends AnyFlatSpec {

  System.gc()

  val spark: SparkSession = ResourceHelper.spark

  import spark.implicits._

  private val emptyDataSet = spark.createDataset(Seq.empty[String]).toDF("text")

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  private val posTagger = getPerceptronModel

  private lazy val dependencyParserConllU = getDependencyParserModelConllU

  private lazy val typedDependencyParserConllU = getTypedDependencyParserModelConllU

  def getPerceptronModel: PerceptronModel = {
    val perceptronTagger = PerceptronModel
      .pretrained()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
    perceptronTagger
  }

  def getDependencyParserModelConllU: DependencyParserModel = {
    val dependencyParser = new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setConllU("src/test/resources/parser/unlabeled/conll-u/train_small.conllu.txt")
      .setNumberOfIterations(15)
      .fit(emptyDataSet)

    val path = "./tmp_dp_ud_model"
    dependencyParser.write.overwrite.save(path)
    DependencyParserModel.read.load(path)
  }

  def getTypedDependencyParserModelConllU: TypedDependencyParserModel = {
    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
      .setNumberOfIterations(3)
      .fit(emptyDataSet)

    val path = "./tmp_tdp_ud_model"
    typedDependencyParser.write.overwrite.save(path)
    TypedDependencyParserModel.read.load(path)
  }

  "A typed dependency parser model (trained with CoNLL-U) with a sentence input" should
    "predict a labeled relationship between words in sentences" taggedAs SlowTest in {
      import SparkAccessor.spark.implicits._

      val pipeline = new Pipeline()
        .setStages(
          Array(
            documentAssembler,
            sentenceDetector,
            tokenizer,
            posTagger,
            dependencyParserConllU,
            typedDependencyParserConllU))

      val model = pipeline.fit(emptyDataSet)
      val typedDependencyParserModel = model.stages.last.asInstanceOf[TypedDependencyParserModel]
      val testDataSet = Seq(
        "So what happened?",
        "It should continue to be defanged.",
        "That too was stopped.").toDS.toDF("text")
      val typedDependencyParserDataFrame = model.transform(testDataSet)
      // typedDependencyParserDataFrame.collect()
//      typedDependencyParserDataFrame.select("labdep").show(false)
      assert(typedDependencyParserModel.isInstanceOf[TypedDependencyParserModel])
      assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

    }

  "A typed dependency parser model" should "A typed dependency parser (trained with CoNLL-U) model with sentences in one row input" taggedAs SlowTest in {
    import SparkAccessor.spark.implicits._
    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          posTagger,
          dependencyParserConllU,
          typedDependencyParserConllU))

    val model = pipeline.fit(emptyDataSet)

    val testDataSet =
      Seq("It should continue to be defanged. " + "That too was stopped.").toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    // typedDependencyParserDataFrame.collect()
//      typedDependencyParserDataFrame.select("labdep").show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser model" should "predict a labeled relationship between words in each sentence" taggedAs SlowTest in {
    import SparkAccessor.spark.implicits._

    val finisher = new Finisher().setInputCols("labdep")

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          posTagger,
          dependencyParserConllU,
          typedDependencyParserConllU,
          finisher))

    val model = pipeline.fit(emptyDataSet)

    val testDataSet =
      Seq("So what happened?", "It should continue to be defanged.", "That too was stopped.").toDS
        .toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    // typedDependencyParserDataFrame.collect()
//      typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

}
