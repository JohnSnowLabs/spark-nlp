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

  private val posTagger = getPerceptronModel //PerceptronModel.pretrained()

  private val dependencyParser = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("src/test/resources/parser/dependency_treebank")
    .setNumberOfIterations(10)

  private val emptyDataSet = PipelineModels.dummyDataset

  private val testDataSet = Seq("I saw a girl with a telescope").toDS.toDF("text")

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
    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      )).fit(emptyDataSet)

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
      )).fit(emptyDataSet)

    val dependencyParserModel = model.stages.last.asInstanceOf[DependencyParserModel]
    val dependencyParserDataFrame = model.transform(testDataSet)
    dependencyParserDataFrame.collect()
    //dependencyParserDataFrame.show(false)
    assert(dependencyParserModel.isInstanceOf[DependencyParserModel])
    assert(dependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A dependency parser with a sentence input" should "predict a relationship between words in the sentence" ignore {
    val dependencyParserModel = DependencyParserModel.read.load("./tmp_dp_model")

    val model = new Pipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParserModel
      )).fit(emptyDataSet)

    val dependencyParserDataFrame = model.transform(testDataSet)
    dependencyParserDataFrame.collect()
    //dependencyParserDataFrame.show(false)
    assert(dependencyParserDataFrame.isInstanceOf[DataFrame])
  }

  "A dependency parser model with a document input" should
    "predict a relationship between words in each sentence" ignore {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val document = "I solved the problem with statistics. " +
      "I saw a girl with a telescope"
    val testDataSet = Seq(document).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A dependency parser model with an input of more than one row" should
    "predict a relationship between words in each sentence" ignore {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val document = Seq(
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow",
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward",
      "I solved the problem with statistics")
    val testDataSet = document.toDS.toDF("text")
    val dependencyParserDataFrame = model.transform(testDataSet)
    dependencyParserDataFrame.collect()
    //dependencyParserDataFrame.show(false)
    assert(dependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A dependency parser model with finisher in its pipeline" should
    "predict a relationship between words in each sentence" in  {
    import SparkAccessor.spark.implicits._

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

    val model = pipeline.fit(emptyDataSet)

//    val document = "The most troublesome report may be the August merchandise trade deficit due out tomorrow. " +
//      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward."

    val errorDocument = "The most troublesome report may be the August merchandise trade deficit due out tomorrow"
    //val document = "I prefer the morning flight through Denver"
    //val document = "Book me the morning flight"
    val testDataSet = Seq(errorDocument).toDS.toDF("text")
    val dependencyParserDataFrame = model.transform(testDataSet)
    //dependencyParserDataFrame.collect()
    dependencyParserDataFrame.select("text","finished_dependency").show(false)
    assert(dependencyParserDataFrame.isInstanceOf[DataFrame])

  }

}
