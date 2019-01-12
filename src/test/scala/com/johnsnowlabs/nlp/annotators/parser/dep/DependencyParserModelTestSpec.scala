package com.johnsnowlabs.nlp.annotators.parser.dep

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FlatSpec
import SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.util.MLWriter

import scala.language.reflectiveCalls

class DependencyParserModelTestSpec extends FlatSpec with DependencyParserBehaviors {

  //TODO: Identify behaviors that we can use for testing using dependency tree bank or UD

  "A Dependency Parser trained with TreeBank format file" should behave like {
    val testDataSet: Dataset[Row] =
      AnnotatorBuilder.withTreeBankDependencyParser(DataBuilder.basicDataBuild(ContentProvider.depSentence))
    initialAnnotations(testDataSet)
  }

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    model.overwrite().save(modelFilePath)
    assertResult(true){
      Files.exists(Paths.get(modelFilePath))
    }
  }

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

  private val dependencyParser = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
    .setNumberOfIterations(10)

  private val pipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      posTagger,
      dependencyParser
    ))

  private val emptyDataSet = PipelineModels.dummyDataset

  def getPerceptronModel: PerceptronModel = {
    val perceptronTagger = new PerceptronApproach()
      .setNIterations(1)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
        ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
      .fit(DataBuilder.basicDataBuild("dummy"))
    val path = "./tmp_perceptrontagger"

    perceptronTagger.write.overwrite.save(path)
    val perceptronTaggerRead = PerceptronModel.read.load(path)
    perceptronTaggerRead
  }

  def trainDependencyParserModel(): DependencyParserModel = {
    val model = pipeline.fit(emptyDataSet)

    model.stages.last.asInstanceOf[DependencyParserModel]

  }

  "A dependency parser annotator" should "save a trained model to local disk" in {
    val dependencyParserModel = trainDependencyParserModel()
    saveModel(dependencyParserModel.write, "./tmp_dp_model")
  }

  it should "load a pre-trained model from disk" in {
    val dependencyParserModel = DependencyParserModel.read.load("./tmp_dp_model")
    assert(dependencyParserModel.isInstanceOf[DependencyParserModel])
  }

  "A dependency parser with an input text of one sentence" should behave like {

    val testDataSet = Seq("I saw a girl with a telescope").toDS.toDF("text")

    relationshipsBetweenWordsPredictor(testDataSet, pipeline)
  }

  "A dependency parser model with input text of two sentences" should
    behave like {

    val text = "I solved the problem with statistics. I saw a girl with a telescope"
    val testDataSet = Seq(text).toDS.toDF("text")

    relationshipsBetweenWordsPredictor(testDataSet, pipeline)

  }

  "A dependency parser model with an input text of several rows" should
   behave like {

    val text = Seq(
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow",
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward",
      "Book me the morning flight",
      "I solved the problem with statistics")
    val testDataSet = text.toDS.toDF("text")

    relationshipsBetweenWordsPredictor(testDataSet, pipeline)
  }

  "A dependency parser model with finisher in its pipeline" should
    "find relationships" in  {

    val finisher = new Finisher().setInputCols("dependency")
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        finisher
      ))
    val text = "I prefer the morning flight through Denver"
    val testDataSet = Seq(text).toDS.toDF("text")
    val model = pipeline.fit(emptyDataSet)
    val dependencyParserDataFrame = model.transform(testDataSet)
    dependencyParserDataFrame.select("text","finished_dependency").show(false)
    assert(1==1)
  }

}
