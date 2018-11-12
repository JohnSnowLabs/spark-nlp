package com.johnsnowlabs.nlp.annotators.parser.dep

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.FlatSpec
import SparkAccessor.spark.implicits._
import org.apache.spark.ml.util.MLWriter

import scala.language.reflectiveCalls

class DependencyParserModelTestSpec extends FlatSpec {

  def fixture = new {
    val df: Dataset[Row] = AnnotatorBuilder.withDependencyParser(DataBuilder.basicDataBuild(ContentProvider.depSentence))
    val dependencies: DataFrame = df.select("dependency")
    val depAnnotations: Seq[Annotation] = dependencies
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
    val tokens: DataFrame = df.select("token")
    val tokenAnnotations: Seq[Annotation] = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
  }

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    model.overwrite().save(modelFilePath)
    assertResult(true){
      Files.exists(Paths.get(modelFilePath))
    }
  }

  "A DependencyParser" should "add annotations" ignore {
    val f = fixture
    assert(f.dependencies.count > 0, "Annotations count should be greater than 0")
  }

  it should "add annotations with the correct annotationType" ignore {
    val f = fixture
    f.depAnnotations.foreach { a =>
      assert(a.annotatorType == AnnotatorType.DEPENDENCY, s"Annotation type should ${AnnotatorType.DEPENDENCY}")
    }
  }

  it should "annotate each token" ignore {
    val f = fixture
    assert(f.tokenAnnotations.size == f.depAnnotations.size, s"Every token should be annotated")
  }

  it should "annotate each word with a head" ignore {
    val f = fixture
    f.depAnnotations.foreach { a =>
      assert(a.result.nonEmpty, s"Result should have a head")
    }
  }

  it should "annotate each word with the correct indexes" ignore {
    val f = fixture
    f.depAnnotations
      .zip(f.tokenAnnotations)
      .foreach { case (dep, token) => assert(dep.begin == token.begin && dep.end == token.end, s"Token and word should have equal indixes") }
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

  private val posTagger = PerceptronModel.pretrained()

  private val dependencyParser = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("src/test/resources/parser/dependency_treebank")
    .setNumberOfIterations(10)
    //.setDependencyTreeBank("/Users/dburbano/tmp/dependency_treebank_small")

  private val emptyDataset = PipelineModels.dummyDataset

  private val testDataset = Seq(
   // "One morning I shot an elephant in my pajamas. How he got into my pajamas I’ll never know."
    //"Set the volume to zero when I 'm in a meeting unless John 's school calls",
    "I solved the problem with statistics"
    //"I saw a girl with a telescope"
    //"The most troublesome report may be the August merchandise trade deficit due out tomorrow.",
    //"MSNBC reported that Facebook bought WhatsApp for 16bn"
  ).toDS.toDF("text")

  def dependencyParserPipeline(): Unit = {

    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      )).fit(emptyDataset)

    val dependencyParserDataset = model.transform(testDataset)
    dependencyParserDataset.show(false)
  }

  def trainDependencyParserModel(): DependencyParserModel = {
    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      )).fit(emptyDataset)

    model.stages.last.asInstanceOf[DependencyParserModel]

  }

  "A dependency parser annotator" should "save a trained model to local disk" in {
    val dependencyParserModel = trainDependencyParserModel()
    saveModel(dependencyParserModel.write, "./tmp/dp_model")
  }

  it should "load a pre-trained model from disk" in {
    val dependencyParserModel = DependencyParserModel.read.load("./tmp/dp_model")
    assert(dependencyParserModel.isInstanceOf[DependencyParserModel])
  }

  "A dependency parser model" should "transform a test dataset" ignore {
    dependencyParserPipeline()
  }

  "A dependency parser with explicit number of iterations" should "train a model" ignore {
    val dependencyParser = new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/dependency_treebank")
      .setNumberOfIterations(5)

    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      )).fit(emptyDataset)

     val smallModel = model.stages.last.asInstanceOf[DependencyParserModel]

    assert(smallModel.isInstanceOf[DependencyParserModel])

  }

  "A dependency parser with a model loaded" should "show results" in {
    val dependencyParserModel = DependencyParserModel.read.load("./tmp/dp_model")

    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParserModel
      )).fit(emptyDataset)

    val result = model.transform(testDataset)
    result.show()
  }

}
