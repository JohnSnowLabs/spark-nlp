package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec

import java.nio.file.{Files, Paths}
import scala.language.reflectiveCalls

class DependencyParserModelTestSpec extends FlatSpec with DependencyParserBehaviors {

  private val spark = ResourceHelper.spark
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

  private val dependencyParserTreeBank = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
    .setNumberOfIterations(10)

  private val dependencyParserConllU = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setConllU("src/test/resources/parser/unlabeled/conll-u/train_small.conllu.txt")
    .setNumberOfIterations(15)

  private val pipelineTreeBank = new Pipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      posTagger,
      dependencyParserTreeBank
    ))

  private val pipelineConllU = new Pipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      posTagger,
      dependencyParserConllU
    ))

  def getPerceptronModel: PerceptronModel = {
    val perceptronTagger = PerceptronModel.pretrained()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
    perceptronTagger
  }

  def trainDependencyParserModelTreeBank(): DependencyParserModel = {
    val model = pipelineTreeBank.fit(emptyDataSet)
    model.stages.last.asInstanceOf[DependencyParserModel]
  }

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    model.overwrite().save(modelFilePath)
    assertResult(true){
      Files.exists(Paths.get(modelFilePath))
    }
  }

  "DependencyParser" should "A Dependency Parser (trained through TreeBank format file)" taggedAs SlowTest in {
    val testDataSet: Dataset[Row] =
      AnnotatorBuilder.withTreeBankDependencyParser(DataBuilder.basicDataBuild(ContentProvider.depSentence))
    initialAnnotations(testDataSet)
  }

  "DependencyParser" should "A dependency parser (trained through TreeBank format file) with an input text of one sentence" taggedAs SlowTest in {
    val testDataSet = Seq("I saw a girl with a telescope").toDS.toDF("text")

    relationshipsBetweenWordsPredictor(testDataSet, pipelineTreeBank)
  }

  "DependencyParser" should "A dependency parser (trained through TreeBank format file) with input text of two sentences" taggedAs SlowTest in {
    behave like {
      val text = "I solved the problem with statistics. I saw a girl with a telescope"
      val testDataSet = Seq(text).toDS.toDF("text")
      relationshipsBetweenWordsPredictor(testDataSet, pipelineTreeBank)
    }

    "DependencyParser" should "A dependency parser (trained through TreeBank format file) with an input text of several rows" taggedAs SlowTest in {
      val text = Seq(
        "The most troublesome report may be the August merchandise trade deficit due out tomorrow",
        "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward",
        "Book me the morning flight",
        "I solved the problem with statistics")
      val testDataSet = text.toDS.toDF("text")
      relationshipsBetweenWordsPredictor(testDataSet, pipelineTreeBank)
    }

    "DependencyParser" should "A dependency parser (trained through Universal Dependencies format file) with an input text of one sentence" taggedAs SlowTest in {
      val testDataSet = Seq(
        "So what happened?",
        "It should continue to be defanged.",
        "That too was stopped."
      ).toDS.toDF("text")
      relationshipsBetweenWordsPredictor(testDataSet, pipelineConllU)
    }

    "DependencyParser" should "A dependency parser (trained through Universal Dependencies format file) with input text of two sentences" taggedAs SlowTest in {
      val text = "I solved the problem with statistics. I saw a girl with a telescope"
      val testDataSet = Seq(text).toDS.toDF("text")

      relationshipsBetweenWordsPredictor(testDataSet, pipelineConllU)
    }

    "DependencyParser" should "A dependency parser (trained through Universal Dependencies format file) with an input text of several rows" taggedAs SlowTest in {
      val text = Seq(
        "The most troublesome report may be the August merchandise trade deficit due out tomorrow",
        "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward",
        "Book me the morning flight",
        "I solved the problem with statistics")
      val testDataSet = text.toDS.toDF("text")

      relationshipsBetweenWordsPredictor(testDataSet, pipelineConllU)
    }

  }

}
